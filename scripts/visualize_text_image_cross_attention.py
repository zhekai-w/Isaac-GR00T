"""
Visualize per-text-token attention to image patches in the Eagle backbone.

For every text token in the prompt, produces a spatial heatmap overlaid on the
input image showing which image patches that token attends to.  Attention is
averaged over the last N layers (configurable) and optionally over heads.

With --box_source, an open-vocabulary detector (OWLv2 / Grounding DINO) locates
*every* instance of the queried object, all boxes are drawn on every subplot, and
each attention map is scored by its concentration ratio (attention mass inside
the boxes divided by their area fraction; 1.0 = uniform, >1 = focused on the
object).  Scores go to a CSV: one `union` row per (frame, token, head) covering
all instances together, plus one row per individual box so you can see which
instance the attention actually landed on.

Usage:
    python scripts/visualize_text_image_cross_attention.py \
        --image_path scripts/images/frame_000000.png \
        --text_query "pick up the red cup"

    python scripts/visualize_text_image_cross_attention.py \
        --image_path ~/work/videos/default_exposure.mp4 \
        --text_query "banana" --frame_stride 100 --per_head

    python scripts/visualize_text_image_cross_attention.py \
        --image_path scripts/images/cups.jpg \
        --text_query "pick up the red cup" --box_query "red cup" \
        --box_source owlv2 --heads 10,11 --per_head
"""

import argparse
import csv
import math
import os
import sys

import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from attention_bbox import (
    box_patch_weights,
    detect_boxes,
    draw_box,
    policy_preprocess,
    score_attention,
    union_patch_weights,
)
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


CSV_FIELDS = [
    "frame", "image_path", "text_query", "box_query", "n_boxes", "box_idx", "label",
    "x0", "y0", "x1", "y1", "det_score",
    "layers", "head", "token", "mass_in", "area_frac", "concentration",
]


def append_csv_rows(csv_path, rows):
    """Append scoring rows, writing the header the first time."""
    if not rows:
        return
    need_header = not os.path.exists(csv_path) or os.path.getsize(csv_path) == 0
    with open(csv_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        if need_header:
            writer.writeheader()
        writer.writerows(rows)
    print(f"Appended {len(rows)} row(s) to {csv_path}")


def make_row(args, frame, image_path, box, det_score, n_boxes, box_idx,
             layer_tag, head, token, scores, label=None):
    row = {
        "frame": frame,
        "image_path": image_path,
        "text_query": args.text_query,
        "box_query": args.box_query or args.text_query,
        "n_boxes": n_boxes,
        "box_idx": box_idx,
        "label": label if label is not None else (args.box_query or args.text_query),
        "x0": "", "y0": "", "x1": "", "y1": "",
        "det_score": "" if det_score is None else f"{det_score:.4f}",
        "layers": layer_tag,
        "head": head,
        "token": token,
        "mass_in": "nan", "area_frac": "nan", "concentration": "nan",
    }
    if box is not None:
        row.update({k: f"{v:.1f}" for k, v in zip(("x0", "y0", "x1", "y1"), box)})
    if scores is not None:
        row.update({k: f"{scores[k]:.6f}" for k in ("mass_in", "area_frac", "concentration")})
    return row


def process_image(args, image_path, processor, model, device, image=None, label=None):
    print(f"\nProcessing: {image_path if label is None else label}")
    if image is None:
        image = Image.open(image_path).convert("RGB")

    # ── ground-truth boxes from an open-vocabulary detector (before any resizing,
    #    so the detector sees the full-resolution frame) ──
    detections, negative_dets = [], {}
    if args.box_source != "none":
        box_query = args.box_query or args.text_query
        negatives = (
            [n.strip() for n in args.box_negatives.split(",")] if args.box_negatives else []
        )
        detections, negative_dets = detect_boxes(
            image, box_query,
            source=args.box_source, model_id=args.box_model,
            threshold=args.box_threshold, device=args.box_device or device,
            nms_iou=args.box_nms, max_boxes=args.max_boxes,
            negatives=negatives, return_negatives=True,
        )
        if not detections:
            print(f"Detector found no '{box_query}' above {args.box_threshold} — scoring skipped.")
        else:
            print(f"{len(detections)} box(es) for '{box_query}':")
            for i, (b, s) in enumerate(detections):
                print(
                    f"  [{i}] [{b[0]:.0f}, {b[1]:.0f}, {b[2]:.0f}, {b[3]:.0f}] "
                    f"(score {s:.3f})"
                )
        for name, dets in negative_dets.items():
            for b, s in dets:
                print(
                    f"  (neg) {name}: [{b[0]:.0f}, {b[1]:.0f}, {b[2]:.0f}, {b[3]:.0f}] "
                    f"(score {s:.3f})"
                )

    if args.policy_preproc:
        # Remap both sets against the *original* image before rebinding it.
        _, negative_dets = policy_preprocess(image, negative_dets)
        image, detections = policy_preprocess(image, detections)
        print(f"Policy preprocessing applied — image now {image.size}, "
              f"{len(detections)} box(es) kept")

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

    assert num_img_tokens % num_tokens_per_tile == 0, (
        f"{num_img_tokens} image tokens is not a multiple of {num_tokens_per_tile} "
        "per tile — the last-tile slice below would be misaligned."
    )
    n_tiles = num_img_tokens // num_tokens_per_tile
    print(
        f"Total image tokens: {num_img_tokens}, Tiles: {n_tiles}, "
        f"Tokens per tile: {num_tokens_per_tile}, Grid: {grid_dim}x{grid_dim}"
    )
    if n_tiles == 1:
        # No thumbnail is appended when the image resolves to a single tile
        # (image_processing_eagle2_5_vl_fast.py:326). That tile is still a plain
        # resize of the whole frame, so the linear pixel->patch map still holds.
        print("Single tile — no separate thumbnail; using that tile.")

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
    if args.layer_range:
        lo, hi = (int(x) for x in args.layer_range.split("-"))
        if not 1 <= lo <= hi <= num_layers:
            raise ValueError(
                f"--layer_range must satisfy 1 <= a <= b <= {num_layers}, got {args.layer_range}"
            )
        start_layer, end_layer = lo - 1, hi
    else:
        start_layer, end_layer = max(0, num_layers - args.layers), num_layers
    num_heads = attentions[0].shape[1]

    # shape: (selected_layers, heads, text_tokens, image_patches)
    cross_attn = []
    for l_idx in range(start_layer, end_layer):
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
    frame = label if label is not None else os.path.splitext(os.path.basename(image_path))[0]
    layer_tag = f"{start_layer + 1}-{end_layer}"

    # ── project the boxes onto the patch grid ──
    # The thumbnail tile is a plain resize of the full frame, so this is a pure
    # linear scale (image_processing_eagle2_5_vl_fast.py:326-329).
    # "union" is the headline score (all instances of the object together); the
    # per-box entries say which instance the attention actually went to.
    box_entries = []  # (box_idx, box, det_score, weights, label)
    target_label = args.box_query or args.text_query
    if detections:
        boxes = [b for b, _ in detections]
        box_entries.append(
            ("union", None, max(s for _, s in detections),
             union_patch_weights(boxes, image.size, grid_dim), target_label)
        )
        for i, (b, s) in enumerate(detections):
            box_entries.append(
                (str(i), b, s, box_patch_weights(b, image.size, grid_dim), target_label)
            )
    # Distractors are scored too: attention leaking onto them is as informative
    # as attention landing on the target.
    for name, dets in negative_dets.items():
        for i, (b, s) in enumerate(dets):
            box_entries.append(
                (f"neg{i}", b, s, box_patch_weights(b, image.size, grid_dim), name)
            )
    rows = []

    def token_scores(attn_2d):
        """(tokens, patches) → {key: [score dict per token]} ({} with no boxes)."""
        return {
            (idx, lab): [
                score_attention(attn_2d[t].numpy(), w) for t in range(attn_2d.shape[0])
            ]
            for idx, _b, _s, w, lab in box_entries
        }

    def plot_scores(scored):
        """Union scores drive the captions; fall back to blanks when undetected."""
        return scored.get(("union", target_label), [None] * len(token_labels))

    def emit(head_tag, scored):
        if not box_entries:
            for t, tok in enumerate(token_labels):
                rows.append(
                    make_row(args, frame, image_path, None, None, 0, "",
                             layer_tag, head_tag, tok, None)
                )
            return
        for idx, box, det_score, _w, lab in box_entries:
            for t, tok in enumerate(token_labels):
                rows.append(
                    make_row(args, frame, image_path, box, det_score,
                             len(detections), idx, layer_tag, head_tag, tok,
                             scored[(idx, lab)][t], label=lab)
                )

    if args.head is not None:
        if args.head < 0 or args.head >= num_heads:
            raise ValueError(f"--head must be in [0, {num_heads - 1}], got {args.head}")
        # Select single head, average across layers → (text_tokens, patches)
        attn_map = cross_attn[:, args.head].mean(dim=0)
        scores = token_scores(attn_map)
        emit(str(args.head), scores)
        _plot_head_averaged(attn_map, token_labels, grid_dim, image, out_stem, args,
                            num_total_layers=num_layers, head_idx=args.head,
                            detections=detections, negative_dets=negative_dets, scores=plot_scores(scores))
    elif args.heads:
        # Parse heads specification (e.g., "9-12" or "0,1,2,3")
        head_indices = []
        for part in args.heads.split(","):
            part = part.strip()
            if "-" in part:
                start, end = map(int, part.split("-"))
                head_indices.extend(range(start, end + 1))
            else:
                head_indices.append(int(part))
        # Validate
        for h in head_indices:
            if h < 0 or h >= num_heads:
                raise ValueError(f"Head index {h} out of range [0, {num_heads - 1}]")
        # Select heads, average across layers → (selected_heads, text_tokens, patches)
        attn_map = cross_attn[:, head_indices].mean(dim=0)
        if args.per_head:
            per_head_scores = [token_scores(attn_map[h]) for h in range(len(head_indices))]
            for h, head_idx in enumerate(head_indices):
                emit(str(head_idx), per_head_scores[h])
            _plot_per_head(attn_map, token_labels, grid_dim, image, out_stem, args,
                           detections=detections, negative_dets=negative_dets,
                           scores=[plot_scores(s) for s in per_head_scores],
                           head_labels=head_indices)
        else:
            # Average across selected heads → (text_tokens, patches)
            attn_map = attn_map.mean(dim=0)
            scores = token_scores(attn_map)
            emit(f"avg[{args.heads}]", scores)
            _plot_head_averaged(attn_map, token_labels, grid_dim, image, out_stem, args,
                                num_total_layers=num_layers, detections=detections, negative_dets=negative_dets,
                                scores=plot_scores(scores))
    elif args.per_head:
        # Average across layers only → (heads, text_tokens, patches)
        attn_map = cross_attn.mean(dim=0)
        head_labels = list(range(attn_map.shape[0]))
        per_head_scores = [token_scores(attn_map[h]) for h in head_labels]
        for h in head_labels:
            emit(str(h), per_head_scores[h])
        _plot_per_head(attn_map, token_labels, grid_dim, image, out_stem, args,
                       detections=detections, negative_dets=negative_dets,
                       scores=[plot_scores(s) for s in per_head_scores],
                       head_labels=head_labels)
    else:
        # Average across both layers and heads → (text_tokens, patches)
        attn_map = cross_attn.mean(dim=(0, 1))
        scores = token_scores(attn_map)
        emit("avg", scores)
        _plot_head_averaged(attn_map, token_labels, grid_dim, image, out_stem, args,
                            num_total_layers=num_layers, detections=detections, negative_dets=negative_dets,
                            scores=plot_scores(scores))

    if args.box_source != "none":
        append_csv_rows(args.csv, rows)
    return rows


def _make_rgba_heatmap(heatmap_grid, img_size, max_alpha=0.55):
    """Normalize heatmap, resize to img_size (W,H), return RGBA array with dynamic alpha."""
    W, H = img_size
    h = heatmap_grid.astype(np.float32)
    h_min, h_max = h.min(), h.max()
    h = (h - h_min) / (h_max - h_min) if h_max > h_min else np.zeros_like(h)
    pil_h = Image.fromarray((h * 255).astype(np.uint8)).resize((W, H), Image.NEAREST)
    h_resized = np.array(pil_h) / 255.0
    rgba = cm.cool(h_resized)  # (H, W, 4)
    rgba[..., 3] = h_resized * max_alpha
    return rgba


def _draw_detections(ax, detections, negative_dets=None, linewidth=1.5, label=True):
    """Draw target boxes in green (numbered to match box_idx) and distractors in red."""
    for name, dets in (negative_dets or {}).items():
        for box, _score in dets:
            draw_box(ax, box, color="red", linewidth=linewidth)
            if label:
                ax.text(
                    box[0], box[1] - 2, name, color="red", fontsize=5,
                    va="bottom", ha="left",
                )
    for i, (box, _score) in enumerate(detections):
        draw_box(ax, box, linewidth=linewidth)
        if label and len(detections) > 1:
            ax.text(
                box[0], box[1] - 2, str(i), color="lime", fontsize=6,
                va="bottom", ha="left",
            )


def _plot_head_averaged(attn_map, token_labels, grid_dim, image, image_path, args,
                        num_total_layers=None, head_idx=None, detections=(),
                        negative_dets=None, scores=None):
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
        rgba = _make_rgba_heatmap(heatmap, image.size)
        ax.imshow(image)
        ax.imshow(rgba)
        _draw_detections(ax, detections, negative_dets)
        max_score = attn_map[i].max().item()
        caption = f"max attention: {max_score:.3f}"
        if scores is not None and scores[i] is not None:
            caption += f" | conc {scores[i]['concentration']:.2f}x"
        ax.text(
            0.5, -0.05, caption, va="top", ha="center",
            transform=ax.transAxes, fontsize=6,
        )
        ax.set_title(f'"{label}"', fontsize=9)
        ax.axis("off")

    # hide unused subplots
    for idx in range(n, nrows * ncols):
        axes[idx // ncols][idx % ncols].axis("off")

    if args.layer_range:
        start, L = (int(x) for x in args.layer_range.split("-"))
    else:
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
    out = os.path.join(args.output_dir, f"{base}_{query_slug}{head_slug}.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out}")


def _plot_per_head(attn_map, token_labels, grid_dim, image, image_path, args,
                   detections=(), negative_dets=None, scores=None, head_labels=None):
    """Grid: rows = heads, cols = text tokens."""
    num_heads, n_tokens, _ = attn_map.shape
    if head_labels is None:
        head_labels = list(range(num_heads))
    fig, axes = plt.subplots(
        num_heads, n_tokens, figsize=(n_tokens * 1.8, num_heads * 1.8), squeeze=False
    )

    for h in range(num_heads):
        for t in range(n_tokens):
            ax = axes[h][t]
            heatmap = attn_map[h, t].numpy().reshape(grid_dim, grid_dim)
            rgba = _make_rgba_heatmap(heatmap, image.size)
            ax.imshow(image)
            ax.imshow(rgba)
            _draw_detections(ax, detections, negative_dets, linewidth=1.0, label=False)
            ax.axis("off")
            if h == 0:
                ax.set_title(f'"{token_labels[t]}"', fontsize=7)
            max_score = attn_map[h, t].max().item()
            caption = f"{max_score:.3f}"
            if scores is not None and scores[h][t] is not None:
                caption += f" | {scores[h][t]['concentration']:.2f}x"
            ax.text(
                0.5, -0.05, caption, va="top", ha="center",
                transform=ax.transAxes, fontsize=6,
            )
            if t == 0:
                ax.text(
                    -0.15, 0.5, f"H{head_labels[h]}", va="center", ha="right",
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
        "--image_path", type=str, default="scripts/images/frame_000000.png",
        help="Single image or directory of images",
    )
    parser.add_argument(
        "--text_query", type=str, required=True,
        help="Text prompt (e.g. 'pick up the red cup')",
    )
    parser.add_argument(
        "--layers", type=int, default=4,
        help="Number of final layers to average over (GR00T keeps 12 LLM layers, "
             "so the default 4 means layers 9-12)",
    )
    parser.add_argument(
        "--layer_range", type=str, default=None,
        help="Average over an explicit inclusive 1-indexed layer range instead, "
             "e.g. '5-8' for the middle layers or '1-12' for all. Overrides --layers.",
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
        "--heads", type=str, default=None,
        help="Average over specific heads only, e.g. '9-12' or '0,1,2,3'. Overrides --per_head.",
    )
    parser.add_argument(
        "--frame_stride", type=int, default=30,
        help="For video input: sample every Nth frame (default 30)",
    )
    parser.add_argument(
        "--box_source", choices=["none", "owlv2", "owlvit", "gdino", "omdet"],
        default="none",
        help="Open-vocab detector used to locate the object and score attention "
             "against it. 'none' (default) keeps the plain visualization.",
    )
    parser.add_argument(
        "--box_query", type=str, default=None,
        help="Detector prompt (default: --text_query). Use a noun phrase like "
             "'red cube' rather than an instruction like 'pick up the red cube'.",
    )
    parser.add_argument(
        "--box_threshold", type=float, default=0.3,
        help="Detection score threshold (default 0.3). Lower it to catch faint "
             "instances, raise it if background clutter is being boxed.",
    )
    parser.add_argument(
        "--box_negatives", type=str, default=None,
        help="Comma-separated labels for the OTHER objects in the scene, scored "
             "in the same pass so they compete for their own boxes "
             "(e.g. 'lemon,banana,apple,bell pepper'). Fixes the common case "
             "where a distractor wins a rare query like 'mango'.",
    )
    parser.add_argument(
        "--box_nms", type=float, default=0.5,
        help="NMS IoU for merging duplicate detections of the same instance "
             "(default 0.5; <=0 disables)",
    )
    parser.add_argument(
        "--max_boxes", type=int, default=0,
        help="Keep at most N boxes, highest score first (default 0 = all)",
    )
    parser.add_argument(
        "--box_model", type=str, default=None,
        help="Override the detector checkpoint id",
    )
    parser.add_argument(
        "--box_device", type=str, default=None,
        help="Device for the detector (default: same as the VLM; 'cpu' saves VRAM)",
    )
    parser.add_argument(
        "--csv", type=str, default=None,
        help="CSV to append scores to (default: <output_dir>/attention_scores.csv)",
    )
    parser.add_argument(
        "--policy_preproc", action="store_true",
        help="Center-crop to 0.95 and resize to 224 (and transform the box the "
             "same way) to match what the deployed policy actually sees",
    )
    parser.add_argument(
        # "--output_dir", type=str, default="attention_outputs/video_frames",
        "--output_dir", type=str, default="attention_outputs/position_info/cube/",
        help="Output directory",
    )
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    if args.csv is None:
        args.csv = os.path.join(args.output_dir, "attention_scores.csv")
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
