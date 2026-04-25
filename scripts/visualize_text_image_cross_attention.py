"""
Visualize per-text-token attention to image patches in the Eagle backbone.

For every text token in the prompt, produces a spatial heatmap overlaid on the
input image showing which image patches that token attends to.  Attention is
averaged over the last N layers (configurable) and optionally over heads.

Usage:
    python scripts/visualize_text_image_cross_attention.py \
        --image_path scripts/frame_000000.png \
        --text_query "pick up the red cup"
"""

import argparse
import math
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from gr00t.model.backbone.eagle_backbone import DEFAULT_EAGLE_PATH
from gr00t.model.backbone.eagle2_hg_model.modeling_eagle2_5_vl import (
    Eagle2_5_VLForConditionalGeneration,
)
from gr00t.model.transforms import build_eagle_processor


def load_model(model_path, device):
    """Load Eagle model with eager attention so attentions are returned."""
    import transformers

    orig_from_pretrained = transformers.AutoConfig.from_pretrained

    def patched_from_pretrained(pretrained_model_name_or_path, **kwargs):
        config = orig_from_pretrained(pretrained_model_name_or_path, **kwargs)
        if hasattr(config, "_attn_implementation"):
            config._attn_implementation = "eager"
        if hasattr(config, "text_config"):
            config.text_config._attn_implementation = "eager"
            if hasattr(config.text_config, "_attn_implementation_autoset"):
                config.text_config._attn_implementation_autoset = False
        return config

    transformers.AutoConfig.from_pretrained = patched_from_pretrained

    try:
        from gr00t.model.gr00t_n1 import GR00T_N1_5

        gr00t_model = GR00T_N1_5.from_pretrained(
            model_path, torch_dtype=torch.bfloat16
        )
        model = gr00t_model.backbone.eagle_model.to(device)
    except Exception as e:
        print(
            f"Could not load as GR00T_N1_5 checkpoint ({e}), "
            "falling back to Eagle2_5_VLForConditionalGeneration..."
        )
        model = Eagle2_5_VLForConditionalGeneration.from_pretrained(
            model_path, torch_dtype=torch.bfloat16, trust_remote_code=True
        ).to(device)
    finally:
        transformers.AutoConfig.from_pretrained = orig_from_pretrained

    model.eval()
    return model


VIDEO_EXTS = (".mp4", ".avi", ".mov", ".mkv", ".webm")


def iter_video_frames(video_path, stride):
    """Yield (pil_image, frame_idx) pairs sampled at the given stride."""
    import cv2
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"Video: {video_path} | total frames: {total} | stride: {stride}")
    idx = 0
    while True:
        ok, frame_bgr = cap.read()
        if not ok:
            break
        if idx % stride == 0:
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            yield Image.fromarray(frame_rgb), idx
        idx += 1
    cap.release()


def process_image(args, image_path, processor, model, device, image=None, label=None):
    print(f"\nProcessing: {image_path if label is None else label}")
    if image is None:
        image = Image.open(image_path).convert("RGB")

    conversation = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": args.text_query},
            ],
        }
    ]

    text_list = [
        processor.apply_chat_template(
            conversation, tokenize=False, add_generation_prompt=True
        )
    ]
    image_inputs, _ = processor.process_vision_info(conversation)

    inputs = processor(
        text=text_list, images=image_inputs, return_tensors="pt", padding=True
    ).to(device, dtype=torch.bfloat16)

    if "pixel_values" in inputs:
        inputs["pixel_values"] = inputs["pixel_values"].to(torch.bfloat16)
    inputs.pop("image_sizes", None)

    # ── forward pass ──
    print("Running forward pass with output_attentions=True ...")
    with torch.no_grad():
        outputs = model(**inputs, output_attentions=True, return_dict=True)

    attentions = outputs.attentions
    if not attentions:
        print("Model did not return attentions.")
        return

    # ── identify image vs text token positions ──
    input_ids = inputs["input_ids"][0]
    IMAGE_TOKEN_ID = model.config.image_token_index
    image_positions = torch.where(input_ids == IMAGE_TOKEN_ID)[0]

    if len(image_positions) == 0:
        print("No image tokens found in input_ids!")
        return

    num_img_tokens = len(image_positions)
    num_tokens_per_tile = getattr(model, "num_image_token", 256)
    grid_dim = int(math.sqrt(num_tokens_per_tile))

    print(
        f"Total image tokens: {num_img_tokens}, "
        f"Tokens per tile: {num_tokens_per_tile}, Grid: {grid_dim}x{grid_dim}"
    )

    # Use the global thumbnail tile (last num_tokens_per_tile image tokens)
    thumbnail_positions = image_positions[-num_tokens_per_tile:]

    # ── identify text token indices and their decoded strings ──
    non_image_mask = input_ids != IMAGE_TOKEN_ID
    all_indices = torch.arange(len(input_ids), device=input_ids.device)
    text_indices = all_indices[non_image_mask]

    # Find token indices that correspond to the user's text query only
    query_token_ids = processor.tokenizer.encode(args.text_query, add_special_tokens=False)
    query_len = len(query_token_ids)

    # Search for the query token sequence in input_ids (last occurrence)
    input_ids_list = input_ids.tolist()
    query_start = None
    for i in range(len(input_ids_list) - query_len, -1, -1):
        if input_ids_list[i : i + query_len] == query_token_ids:
            query_start = i
            break

    token_labels = []
    token_indices = []
    if query_start is not None:
        for offset in range(query_len):
            idx = query_start + offset
            decoded = processor.tokenizer.decode([input_ids[idx]])
            stripped = decoded.strip()
            if stripped:
                token_labels.append(stripped)
                token_indices.append(idx)
    else:
        # Fallback: match individual tokens from the query
        for idx in text_indices:
            decoded = processor.tokenizer.decode([input_ids[idx]])
            stripped = decoded.strip()
            if stripped and stripped.lower() in args.text_query.lower():
                token_labels.append(stripped)
                token_indices.append(idx.item())

    if not token_labels:
        print("No non-empty text tokens found.")
        return

    print(f"Text tokens ({len(token_labels)}): {token_labels}")

    # ── extract & average attention scores ──
    num_layers = len(attentions)
    start_layer = max(0, num_layers - args.layers)
    num_heads = attentions[0].shape[1]

    # shape: (selected_layers, heads, text_tokens, image_patches)
    cross_attn = []
    for l_idx in range(start_layer, num_layers):
        layer_attn = attentions[l_idx][0]  # (heads, seq, seq)
        # For each text token, get its attention to thumbnail image patches
        scores = []
        for t_idx in token_indices:
            head_scores = layer_attn[:, t_idx, thumbnail_positions]  # (heads, patches)
            scores.append(head_scores)
        scores = torch.stack(scores, dim=1)  # (heads, text_tokens, patches)
        cross_attn.append(scores)

    cross_attn = torch.stack(cross_attn, dim=0)  # (layers, heads, text_tokens, patches)
    cross_attn = cross_attn.float().cpu()

    out_stem = label if label is not None else image_path
    if args.head is not None:
        if args.head < 0 or args.head >= num_heads:
            raise ValueError(f"--head must be in [0, {num_heads - 1}], got {args.head}")
        # Select single head, average across layers → (text_tokens, patches)
        attn_map = cross_attn[:, args.head].mean(dim=0)
        _plot_head_averaged(attn_map, token_labels, grid_dim, image, out_stem, args,
                            num_total_layers=num_layers, head_idx=args.head)
    elif args.per_head:
        # Average across layers only → (heads, text_tokens, patches)
        attn_map = cross_attn.mean(dim=0)
        _plot_per_head(attn_map, token_labels, grid_dim, image, out_stem, args)
    else:
        # Average across both layers and heads → (text_tokens, patches)
        attn_map = cross_attn.mean(dim=(0, 1))
        _plot_head_averaged(attn_map, token_labels, grid_dim, image, out_stem, args,
                            num_total_layers=num_layers)


def _plot_head_averaged(attn_map, token_labels, grid_dim, image, image_path, args,
                        num_total_layers=None, head_idx=None):
    """One subplot per text token, head-averaged."""
    n = len(token_labels)
    ncols = min(n, 8)
    nrows = math.ceil(n / ncols)

    fig, axes = plt.subplots(
        nrows, ncols, figsize=(ncols * 2.5, nrows * 2.5), squeeze=False
    )

    for i, (label, ax) in enumerate(
        zip(token_labels, [axes[r][c] for r in range(nrows) for c in range(ncols)])
    ):
        heatmap = attn_map[i].numpy().reshape(grid_dim, grid_dim)
        ax.imshow(image, extent=[0, grid_dim, grid_dim, 0])
        ax.imshow(heatmap, cmap="jet", interpolation="nearest", alpha=0.55,
                  extent=[0, grid_dim, grid_dim, 0])
        ax.set_title(f'"{label}"', fontsize=9)
        ax.axis("off")

    # hide unused subplots
    for idx in range(n, nrows * ncols):
        axes[idx // ncols][idx % ncols].axis("off")

    L = num_total_layers or args.layers
    start = max(1, L - args.layers + 1)
    head_tag = f"head {head_idx}" if head_idx is not None else "head avg"
    plt.suptitle(
        f"Text\u2192Image attention (layers {start}\u2013{L}, {head_tag})",
        fontsize=11,
    )
    plt.tight_layout()

    base = os.path.splitext(os.path.basename(image_path))[0]
    query_slug = args.text_query.replace(" ", "_")
    head_slug = f"_head{head_idx}" if head_idx is not None else ""
    out = os.path.join(args.output_dir, f"cross_attn_{base}_{query_slug}{head_slug}.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out}")


def _plot_per_head(attn_map, token_labels, grid_dim, image, image_path, args):
    """Grid: rows = heads, cols = text tokens."""
    num_heads, n_tokens, _ = attn_map.shape
    fig, axes = plt.subplots(
        num_heads, n_tokens, figsize=(n_tokens * 1.8, num_heads * 1.8), squeeze=False
    )

    for h in range(num_heads):
        for t in range(n_tokens):
            ax = axes[h][t]
            heatmap = attn_map[h, t].numpy().reshape(grid_dim, grid_dim)
            ax.imshow(image, extent=[0, grid_dim, grid_dim, 0])
            ax.imshow(heatmap, cmap="jet", interpolation="nearest", alpha=0.55,
                      extent=[0, grid_dim, grid_dim, 0])
            ax.axis("off")
            if h == 0:
                ax.set_title(f'"{token_labels[t]}"', fontsize=7)
            if t == 0:
                ax.text(
                    -0.15, 0.5, f"H{h}", va="center", ha="right",
                    transform=ax.transAxes, fontsize=9, fontweight="bold",
                )

    plt.suptitle("Text→Image attention per head (layer avg)", fontsize=11)
    plt.tight_layout()

    base = os.path.splitext(os.path.basename(image_path))[0]
    query_slug = args.text_query.replace(" ", "_")
    out = os.path.join(args.output_dir, f"cross_attn_perhead_{base}_{query_slug}.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out}")


def main():
    parser = argparse.ArgumentParser(
        description="Visualize per-text-token attention to image patches"
    )
    parser.add_argument(
        "--model_path", type=str, default="nvidia/GR00T-N1.5-3B",
        help="Path to model checkpoint",
    )
    parser.add_argument(
        "--image_path", type=str, default="scripts/frame_000000.png",
        help="Single image or directory of images",
    )
    parser.add_argument(
        "--text_query", type=str, required=True,
        help="Text prompt (e.g. 'pick up the red cup')",
    )
    parser.add_argument(
        "--layers", type=int, default=4,
        help="Number of final layers to average over",
    )
    parser.add_argument(
        "--per_head", action="store_true",
        help="Show per-head subplots (rows=heads, cols=tokens) instead of head-averaged",
    )
    parser.add_argument(
        "--head", type=int, default=None,
        help="Visualize only this head index (overrides --per_head). Head-averaged if omitted.",
    )
    parser.add_argument(
        "--frame_stride", type=int, default=30,
        help="For video input: sample every Nth frame (default 30)",
    )
    parser.add_argument(
        # "--output_dir", type=str, default="attention_outputs/video_frames",
        "--output_dir", type=str, default="attention_outputs/position_info",
        help="Output directory",
    )
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"Loading processor from {DEFAULT_EAGLE_PATH} ...")
    processor = build_eagle_processor(DEFAULT_EAGLE_PATH)

    print(f"Loading model from {args.model_path} ...")
    model = load_model(args.model_path, device)

    if os.path.isdir(args.image_path):
        import glob

        image_files = sorted(
            glob.glob(os.path.join(args.image_path, "*.[pP][nN][gG]"))
            + glob.glob(os.path.join(args.image_path, "*.[jJ][pP][gG]"))
        )
        if not image_files:
            print(f"No images found in {args.image_path}")
            return
        print(f"Found {len(image_files)} image(s).")
        for img in image_files:
            process_image(args, img, processor, model, device)
    elif args.image_path.lower().endswith(VIDEO_EXTS):
        video_stem = os.path.splitext(os.path.basename(args.image_path))[0]
        for frame_img, frame_idx in iter_video_frames(args.image_path, args.frame_stride):
            label = f"{video_stem}_f{frame_idx:06d}"
            process_image(
                args, args.image_path, processor, model, device,
                image=frame_img, label=label,
            )
    else:
        process_image(args, args.image_path, processor, model, device)


if __name__ == "__main__":
    main()
