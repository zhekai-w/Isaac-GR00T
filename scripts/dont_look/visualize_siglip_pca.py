# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""
Visualize SigLIP vision encoder output tokens via PCA projected to RGB,
and optionally visualize text-guided activation heatmaps.

Usage:
    # PCA visualization only
    python scripts/visualize_siglip_pca.py --images /path/to/img.jpg

    # Text-guided heatmap (shows which patches activate for each query)
    python scripts/visualize_siglip_pca.py --images img.jpg --text "a robot arm" "a table"

    # Save output
    python scripts/visualize_siglip_pca.py --images img1.jpg img2.jpg --output out.png

    # Use 224px model (faster)
    python scripts/visualize_siglip_pca.py --images img.jpg --model google/siglip-so400m-patch14-224

    # From local GR00T Eagle2 checkpoint (PCA only; text mode needs full SigLIP)
    python scripts/visualize_siglip_pca.py --images img.jpg --model /path/to/eagle2_checkpoint

    # Visualize an intermediate layer
    python scripts/visualize_siglip_pca.py --images img.jpg --layer 14
"""

import json
import math
import os
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from transformers import AutoTokenizer, SiglipImageProcessor, SiglipModel, SiglipVisionModel


@dataclass
class Args:
    images: List[str]
    """One or more input image paths."""

    model: str = "google/siglip-so400m-patch14-384"
    """HuggingFace model name or local path. Supports:
       - Any SiglipVisionModel HF name (e.g. google/siglip-so400m-patch14-384)
       - Path to a local GR00T Eagle2 checkpoint directory (extracts embedded SigLIP)
    """

    text: Optional[List[str]] = None
    """One or more text queries to visualize as patch similarity heatmaps.
    Requires a full SigLIP model (not Eagle2 checkpoint). Example:
      --text "a robot gripper" "a table surface"
    """

    output: Optional[str] = None
    """If set, save the plot to this path instead of displaying interactively."""

    layer: int = -1
    """Which layer to visualize. -1 = last_hidden_state, otherwise hidden_states[layer]."""

    device: str = "cuda"
    """Device to run inference on ('cuda' or 'cpu')."""


def _is_eagle2_checkpoint(path: str) -> bool:
    config_path = os.path.join(path, "config.json")
    if not os.path.isfile(config_path):
        return False
    with open(config_path) as f:
        cfg = json.load(f)
    return cfg.get("model_type") == "eagle_2_5_vl"


def load_siglip_model(
    model_path: str, device: str, need_text: bool = False
) -> Tuple[object, SiglipImageProcessor, Optional[AutoTokenizer]]:
    """Load SigLIP vision model (and optionally text encoder) plus processor.

    Returns:
        model: SiglipModel (if need_text) or SiglipVisionModel
        processor: SiglipImageProcessor
        tokenizer: AutoTokenizer if need_text, else None
    """
    if os.path.isdir(model_path) and _is_eagle2_checkpoint(model_path):
        if need_text:
            raise ValueError(
                "Text mode requires a full SigLIP model. Eagle2 checkpoints only "
                "contain vision encoder weights. Use a HuggingFace SigLIP model, e.g.:\n"
                "  --model google/siglip-so400m-patch14-384"
            )
        print("Detected Eagle2/GR00T checkpoint — extracting embedded SigLIP weights.")
        vision_model, processor = _load_siglip_from_eagle2(model_path, device)
        return vision_model, processor, None

    if need_text:
        print(f"Loading full SigLIP (vision + text) from: {model_path}")
        model = SiglipModel.from_pretrained(model_path)
        model = model.to(device).eval()
        tokenizer = AutoTokenizer.from_pretrained(model_path)
    else:
        print(f"Loading SigLIP vision model from: {model_path}")
        model = SiglipVisionModel.from_pretrained(model_path)
        model = model.to(device).eval()
        tokenizer = None

    processor = SiglipImageProcessor.from_pretrained(model_path)
    return model, processor, tokenizer


def _load_siglip_from_eagle2(
    checkpoint_path: str, device: str
) -> Tuple[SiglipVisionModel, SiglipImageProcessor]:  # type: ignore[return]
    """Extract SigLIP from a local Eagle2 checkpoint without loading the full LLM."""
    from transformers import AutoConfig

    config = AutoConfig.from_pretrained(checkpoint_path, trust_remote_code=True)
    vision_cfg = config.vision_config

    # Override flash-attn to avoid requiring flash-attn package
    vision_cfg._attn_implementation = "sdpa"

    model = SiglipVisionModel(vision_cfg)

    # Find and load only vision_model.* weights from the checkpoint shards
    import glob

    weight_files = sorted(glob.glob(os.path.join(checkpoint_path, "*.safetensors")))
    if not weight_files:
        weight_files = sorted(glob.glob(os.path.join(checkpoint_path, "*.bin")))
    if not weight_files:
        raise FileNotFoundError(f"No weight files found in {checkpoint_path}")

    vision_state_dict = {}
    prefix = "vision_model."
    for wf in weight_files:
        if wf.endswith(".safetensors"):
            from safetensors.torch import load_file

            sd = load_file(wf)
        else:
            sd = torch.load(wf, map_location="cpu", weights_only=True)
        for k, v in sd.items():
            if k.startswith(prefix):
                vision_state_dict[k[len(prefix) :]] = v

    if not vision_state_dict:
        raise RuntimeError(
            "No 'vision_model.*' keys found in checkpoint. "
            "Check that this is an Eagle2 checkpoint with an embedded SigLIP."
        )

    missing, unexpected = model.load_state_dict(vision_state_dict, strict=False)
    if missing:
        print(f"  Warning: missing keys ({len(missing)}): {missing[:5]}{'...' if len(missing)>5 else ''}")

    model = model.to(device).eval()

    # Build processor from vision config parameters
    img_size = vision_cfg.image_size
    processor = SiglipImageProcessor(
        size={"height": img_size, "width": img_size},
        image_mean=[0.5, 0.5, 0.5],
        image_std=[0.5, 0.5, 0.5],
        do_resize=True,
        do_normalize=True,
    )
    return model, processor


def preprocess_image(
    image_path: str, processor: SiglipImageProcessor, device: str
) -> Tuple[torch.Tensor, Image.Image, Tuple[int, int]]:
    """Load and preprocess an image for SigLIP.

    Returns:
        pixel_values: [1, 3, H, W] tensor on device
        pil_image: original PIL image (for display)
        orig_size: (H, W) of original image
    """
    pil_image = Image.open(image_path).convert("RGB")
    orig_size = (pil_image.height, pil_image.width)
    inputs = processor(images=pil_image, return_tensors="pt")
    pixel_values = inputs["pixel_values"].to(device)
    return pixel_values, pil_image, orig_size


@torch.no_grad()
def extract_patch_tokens(
    model: SiglipVisionModel,
    pixel_values: torch.Tensor,
    layer: int = -1,
) -> np.ndarray:
    """Run SigLIP and return patch token features as float32 numpy array [N, C]."""
    if layer == -1:
        outputs = model(pixel_values=pixel_values, return_dict=True)
        tokens = outputs.last_hidden_state  # [1, N, C]
    else:
        outputs = model(
            pixel_values=pixel_values,
            output_hidden_states=True,
            return_dict=True,
        )
        tokens = outputs.hidden_states[layer]  # [1, N, C]

    return tokens.squeeze(0).float().cpu().numpy()  # [N, C]


def compute_pca_rgb(tokens: np.ndarray) -> np.ndarray:
    """Project [N, C] token features to [N, 3] RGB via PCA.

    Uses torch.pca_lowrank (randomized SVD). Each component is independently
    normalized to [0, 1] for maximum color contrast.
    """
    X = torch.from_numpy(tokens)  # [N, C]
    X = X - X.mean(dim=0, keepdim=True)  # center
    U, S, _V = torch.pca_lowrank(X, q=3)
    components = U[:, :3] * S[:3]  # [N, 3] PCA scores

    # Normalize each component independently to [0, 1]
    lo = components.min(dim=0).values
    hi = components.max(dim=0).values
    components = (components - lo) / (hi - lo + 1e-8)
    return components.numpy().astype(np.float32)  # [N, 3]


def _encode_text(model: SiglipModel, tokenizer: AutoTokenizer, query: str, device: str) -> torch.Tensor:
    """Encode a text query to a normalized [1, C] feature vector."""
    inputs = tokenizer(
        text=[query],
        return_tensors="pt",
        padding="max_length",
        truncation=True,
    ).to(device)
    with torch.no_grad():
        text_out = model.get_text_features(**inputs)
        text_feat = text_out.pooler_output if hasattr(text_out, "pooler_output") else text_out[1]
    return F.normalize(text_feat.float(), dim=-1)  # [1, C]


def compute_text_gradcam(
    model: SiglipModel,
    tokenizer: AutoTokenizer,
    pixel_values: torch.Tensor,
    text_queries: List[str],
    device: str,
) -> List[np.ndarray]:
    """GradCAM heatmap: which patches drove the image-text similarity score.

    SigLIP is trained at the global (pooled) level, so raw patch-text cosine
    similarity is uninformative. GradCAM backprops the similarity score through
    the vision encoder to identify spatially relevant patches.

    Returns:
        List of [N] float32 arrays in [0, 1], one per text query.
    """
    # Capture the last encoder layer's activations and gradients
    activations: dict = {}
    gradients: dict = {}

    last_layer = model.vision_model.encoder.layers[-1]

    def fwd_hook(_module, _inp, output):
        # encoder layer output is a tuple; first element is hidden states [1, N, C]
        activations["feat"] = output[0]

    def bwd_hook(_module, _grad_in, grad_out):
        gradients["feat"] = grad_out[0]  # [1, N, C]

    h_fwd = last_layer.register_forward_hook(fwd_hook)
    h_bwd = last_layer.register_full_backward_hook(bwd_hook)

    heatmaps = []
    try:
        for query in text_queries:
            text_norm = _encode_text(model, tokenizer, query, device)  # [1, C]

            model.zero_grad()
            # Forward with grad enabled to get image pooled features
            vision_out = model.vision_model(pixel_values=pixel_values, return_dict=True)
            image_feat = F.normalize(vision_out.pooler_output.float(), dim=-1)  # [1, C]

            # Scalar similarity score → backprop
            sim = (image_feat * text_norm).sum()
            sim.backward()

            act = activations["feat"].detach().float()   # [1, N, C]
            grad = gradients["feat"].detach().float()    # [1, N, C]

            # GradCAM: weight channels by mean gradient, sum, ReLU
            weights = grad.mean(dim=-1, keepdim=True)    # [1, N, 1]
            cam = (weights * act).sum(dim=-1).squeeze(0) # [N]
            cam = F.relu(cam).cpu()

            cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
            heatmaps.append(cam.numpy().astype(np.float32))
    finally:
        h_fwd.remove()
        h_bwd.remove()

    return heatmaps


def heatmap_to_image(
    scores: np.ndarray,  # [N] in [0, 1]
    target_h: int,
    target_w: int,
    orig_image: Image.Image,
    alpha: float = 0.5,
    cmap: str = "hot",
) -> np.ndarray:
    """Render patch similarity scores as a colored heatmap overlaid on the original image.

    Returns uint8 [H, W, 3].
    """
    N = scores.shape[0]
    grid_size = int(math.isqrt(N))
    assert grid_size * grid_size == N

    score_grid = scores.reshape(grid_size, grid_size)
    colormap = matplotlib.colormaps[cmap]
    colored = colormap(score_grid)[:, :, :3]  # [g, g, 3] float in [0,1]
    colored_pil = Image.fromarray((colored * 255).astype(np.uint8))
    colored_pil = colored_pil.resize((target_w, target_h), Image.BILINEAR)

    # Blend with original image
    orig_arr = np.array(orig_image.resize((target_w, target_h))).astype(np.float32)
    heat_arr = np.array(colored_pil).astype(np.float32)
    blended = ((1 - alpha) * orig_arr + alpha * heat_arr).clip(0, 255).astype(np.uint8)
    return blended


def tokens_to_rgb_image(
    pca_tokens: np.ndarray, target_h: int, target_w: int
) -> np.ndarray:
    """Reshape [N, 3] PCA tokens to spatial RGB image upsampled to (target_h, target_w).

    Returns uint8 numpy array [H, W, 3].
    """
    N = pca_tokens.shape[0]
    grid_size = int(math.isqrt(N))
    assert grid_size * grid_size == N, (
        f"N={N} patches is not a perfect square. "
        f"Got grid_size={grid_size}, expected {grid_size**2}."
    )

    pca_grid = pca_tokens.reshape(grid_size, grid_size, 3)
    pca_pil = Image.fromarray((pca_grid * 255).astype(np.uint8))
    pca_pil = pca_pil.resize((target_w, target_h), Image.BILINEAR)
    return np.array(pca_pil)


def visualize_results(
    results: List[Tuple[Image.Image, np.ndarray, str, List[Tuple[str, np.ndarray]]]],
    output_path: Optional[str],
):
    """Render original | PCA | [text heatmaps...] for each image.

    Each entry in results: (orig_pil, pca_arr, label, [(query, heatmap_arr), ...])
    """
    n_images = len(results)
    n_text = len(results[0][3]) if results else 0
    n_cols = 2 + n_text  # original + PCA + one per text query

    fig, axes = plt.subplots(n_images, n_cols, figsize=(5 * n_cols, 5 * n_images))
    if n_images == 1:
        axes = axes[np.newaxis, :]
    if n_cols == 1:
        axes = axes[:, np.newaxis]

    for i, (orig_pil, pca_arr, label, text_maps) in enumerate(results):
        axes[i, 0].imshow(np.array(orig_pil))
        axes[i, 0].set_title(f"{label}\nOriginal")
        axes[i, 0].axis("off")

        axes[i, 1].imshow(pca_arr)
        axes[i, 1].set_title("SigLIP PCA (RGB)")
        axes[i, 1].axis("off")

        for j, (query, heatmap) in enumerate(text_maps):
            axes[i, 2 + j].imshow(heatmap)
            axes[i, 2 + j].set_title(f'"{query}"')
            axes[i, 2 + j].axis("off")

    plt.tight_layout()

    if output_path:
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
        print(f"Saved to: {output_path}")
    else:
        plt.show()


def main(args: Args):
    if args.device == "cuda" and not torch.cuda.is_available():
        print("CUDA not available, falling back to CPU.")
        args.device = "cpu"

    need_text = bool(args.text)
    model, processor, tokenizer = load_siglip_model(args.model, args.device, need_text)

    # For text mode, extract the vision sub-model for patch tokens
    vision_model = model.vision_model if isinstance(model, SiglipModel) else model

    results = []
    for img_path in args.images:
        print(f"Processing: {img_path}")
        pixel_values, pil_image, (orig_h, orig_w) = preprocess_image(
            img_path, processor, args.device
        )
        tokens = extract_patch_tokens(vision_model, pixel_values, layer=args.layer)
        print(f"  Token shape: {tokens.shape}  (N={tokens.shape[0]}, C={tokens.shape[1]})")

        pca_tokens = compute_pca_rgb(tokens)
        pca_image = tokens_to_rgb_image(pca_tokens, orig_h, orig_w)

        text_maps = []
        if need_text:
            print(f"  Computing GradCAM for: {args.text}")
            heatmaps = compute_text_gradcam(
                model, tokenizer, pixel_values, args.text, args.device
            )
            for query, heatmap in zip(args.text, heatmaps):
                overlay = heatmap_to_image(heatmap, orig_h, orig_w, pil_image)
                text_maps.append((query, overlay))

        label = os.path.basename(img_path)
        results.append((pil_image, pca_image, label, text_maps))

    visualize_results(results, args.output)


if __name__ == "__main__":
    import tyro

    main(tyro.cli(Args))
