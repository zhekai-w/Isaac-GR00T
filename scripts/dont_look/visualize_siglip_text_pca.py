# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""
Visualize SigLIP vision and text tokens via PCA and cross-modal activation heatmaps.

This script allows you to:
1. Visualize image patch tokens as RGB via PCA.
2. Visualize text token and pooled embeddings as a scatter plot via PCA.
3. See how different text inputs "activate" different parts of the image via similarity heatmaps.

Usage:
    python scripts/visualize_siglip_text_pca.py --images img.jpg --texts "a robot arm" "a table"
"""

import json
import math
import os
from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Dict

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from transformers import AutoTokenizer, SiglipImageProcessor, SiglipModel, SiglipVisionModel

@dataclass
class Args:
    images: List[str] = field(default_factory=list)
    """One or more input image paths."""

    texts: List[str] = field(default_factory=list)
    """One or more text queries to visualize."""

    model: str = "google/siglip-so400m-patch14-384"
    """HuggingFace model name or local path (e.g. Eagle2 checkpoint)."""

    output: Optional[str] = None
    """Save output to this path."""

    layer: int = -1
    """Layer to extract tokens from. -1 for last hidden state."""

    device: str = "cuda"
    """Device: 'cuda' or 'cpu'."""

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
    if os.path.isdir(model_path) and _is_eagle2_checkpoint(model_path):
        if need_text:
            raise ValueError("Text mode requires full SigLIP. Eagle2 checkpoints only contain vision weights.")
        vision_model, processor = _load_siglip_from_eagle2(model_path, device)
        return vision_model, processor, None

    if need_text:
        model = SiglipModel.from_pretrained(model_path).to(device).eval()
        tokenizer = AutoTokenizer.from_pretrained(model_path)
    else:
        model = SiglipVisionModel.from_pretrained(model_path).to(device).eval()
        tokenizer = None

    processor = SiglipImageProcessor.from_pretrained(model_path)
    return model, processor, tokenizer

def _load_siglip_from_eagle2(checkpoint_path: str, device: str) -> Tuple[SiglipVisionModel, SiglipImageProcessor]:
    from transformers import AutoConfig
    config = AutoConfig.from_pretrained(checkpoint_path, trust_remote_code=True)
    vision_cfg = config.vision_config
    vision_cfg._attn_implementation = "sdpa"
    model = SiglipVisionModel(vision_cfg)
    
    import glob
    weight_files = sorted(glob.glob(os.path.join(checkpoint_path, "*.safetensors"))) or \
                   sorted(glob.glob(os.path.join(checkpoint_path, "*.bin")))
    if not weight_files:
        raise FileNotFoundError(f"No weights in {checkpoint_path}")

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
                vision_state_dict[k[len(prefix):]] = v
    
    model.load_state_dict(vision_state_dict, strict=False)
    model = model.to(device).eval()
    processor = SiglipImageProcessor(
        size={"height": vision_cfg.image_size, "width": vision_cfg.image_size},
        image_mean=[0.5, 0.5, 0.5], image_std=[0.5, 0.5, 0.5],
        do_resize=True, do_normalize=True,
    )
    return model, processor

def preprocess_image(image_path: str, processor: SiglipImageProcessor, device: str):
    pil_image = Image.open(image_path).convert("RGB")
    orig_size = (pil_image.height, pil_image.width)
    inputs = processor(images=pil_image, return_tensors="pt")
    pixel_values = inputs["pixel_values"].to(device)
    return pixel_values, pil_image, orig_size

@torch.no_grad()
def extract_patch_tokens(model: SiglipVisionModel, pixel_values: torch.Tensor, layer: int = -1):
    if layer == -1:
        tokens = model(pixel_values=pixel_values).last_hidden_state
    else:
        tokens = model(pixel_values=pixel_values, output_hidden_states=True).hidden_states[layer]
    return tokens.squeeze(0).float().cpu().numpy()

def compute_pca_rgb(tokens: np.ndarray) -> np.ndarray:
    X = torch.from_numpy(tokens)
    X = X - X.mean(dim=0, keepdim=True)
    U, S, _ = torch.pca_lowrank(X, q=3)
    components = U[:, :3] * S[:3]
    lo, hi = components.min(dim=0).values, components.max(dim=0).values
    return ((components - lo) / (hi - lo + 1e-8)).numpy().astype(np.float32)

@torch.no_grad()
def extract_text_features(model: SiglipModel, tokenizer: AutoTokenizer, texts: List[str], device: str, layer: int = -1):
    """Extracts both pooled and token-level features for the given texts."""
    results = []
    for text in texts:
        inputs = tokenizer(text, return_tensors="pt", padding="max_length", truncation=True).to(device)
        outputs = model.get_text_features(**inputs)
        
        # Pooled embedding
        pooled = outputs.pooler_output if hasattr(outputs, "pooler_output") else outputs[1]
        
        # Token-level embeddings (from text_model)
        text_model_out = model.text_model(**inputs, output_hidden_states=True)
        tokens = text_model_out.last_hidden_state if layer == -1 else text_model_out.hidden_states[layer]
        
        # Decode tokens to actual words for labeling
        token_words = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])
        
        results.append({
            "text": text,
            "pooled": pooled.squeeze(0).float().cpu().numpy(),
            "tokens": tokens.squeeze(0).float().cpu().numpy(),
            "words": token_words
        })
    return results

@torch.no_grad()
def compute_activation_heatmaps(model: SiglipModel, tokenizer: AutoTokenizer, 
                               patch_tokens: np.ndarray, texts: List[str], device: str):
    patch_t = F.normalize(torch.from_numpy(patch_tokens).to(device).float(), dim=-1)
    heatmaps = []
    for text in texts:
        inputs = tokenizer(text, return_tensors="pt", padding="max_length", truncation=True).to(device)
        text_feat = model.get_text_features(**inputs)
        text_feat = text_feat.pooler_output if hasattr(text_feat, "pooler_output") else text_feat[1]
        text_norm = F.normalize(text_feat.float(), dim=-1)
        
        sim = (patch_t @ text_norm.T).squeeze(-1).cpu()
        sim = (sim - sim.min()) / (sim.max() - sim.min() + 1e-8)
        heatmaps.append(sim.numpy().astype(np.float32))
    return heatmaps

def heatmap_to_image(scores: np.ndarray, target_h: int, target_w: int, orig_image: Image.Image, alpha=0.5):
    N = scores.shape[0]
    grid_size = int(math.isqrt(N))
    score_grid = scores.reshape(grid_size, grid_size)
    cmap = matplotlib.colormaps["hot"]
    colored = cmap(score_grid)[:, :, :3]
    colored_pil = Image.fromarray((colored * 255).astype(np.uint8)).resize((target_w, target_h), Image.BILINEAR)
    orig_arr = np.array(orig_image.resize((target_w, target_h))).astype(np.float32)
    heat_arr = np.array(colored_pil).astype(np.float32)
    return ((1 - alpha) * orig_arr + alpha * heat_arr).clip(0, 255).astype(np.uint8)

def visualize_all(images_data, text_data, model_name, output_path):
    # images_data: List of (orig_pil, pca_image, patch_tokens, orig_size)
    # text_data: List of {text, pooled, tokens, words}
    
    n_imgs = len(images_data)
    n_texts = len(text_data)
    
    # We'll create a large figure for each image
    for img_idx, (orig_pil, pca_img, patch_tokens, (oh, ow)) in enumerate(images_data):
        n_cols = 2 + n_texts # Original, PCA, Heatmap per text
        fig, axes = plt.subplots(1, n_cols, figsize=(5 * n_cols, 5))
        if n_cols == 1: axes = [axes]
        
        axes[0].imshow(orig_pil)
        axes[0].set_title("Original")
        axes[0].axis("off")
        
        axes[1].imshow(pca_img)
        axes[1].set_title("Vision PCA (RGB)")
        axes[1].axis("off")
        
        # Compute and show heatmaps for each text
        # Note: We need the full model and tokenizer here, but we can pass the pre-computed heatmaps
        # For simplicity in this layout, I'll assume the main loop handles heatmap generation
        
    # For Text PCA scatter plot
    plt.figure(figsize=(10, 8))
    all_text_tokens = []
    all_text_labels = []
    all_pooled = []
    
    for td in text_data:
        all_text_tokens.append(td["tokens"])
        all_text_labels.extend(td["words"])
        all_pooled.append(td["pooled"])
        
    # PCA for text
    tokens_concat = np.concatenate(all_text_tokens)
    pca_tokens = compute_pca_rgb(tokens_concat)
    
    plt.scatter(pca_tokens[:, 0], pca_tokens[:, 1], c=pca_tokens[:, 2], cmap='viridis', alpha=0.6, s=20)
    for i, label in enumerate(all_text_labels):
        if len(label) < 10: # Only label short tokens
            plt.text(pca_tokens[i, 0], pca_tokens[i, 1], label, fontsize=8)
            
    plt.title("Text Token PCA (Projected to 2D)")
    plt.colorbar(label="PCA Component 3")
    plt.axis("off")

def main(args: Args):
    if args.device == "cuda" and not torch.cuda.is_available():
        args.device = "cpu"
    
    need_text = bool(args.texts)
    model, processor, tokenizer = load_siglip_model(args.model, args.device, need_text)
    vision_model = model.vision_model if isinstance(model, SiglipModel) else model
    
    text_features = []
    if need_text:
        text_features = extract_text_features(model, tokenizer, args.texts, args.device, args.layer)
        
    # Result visualization
    for img_path in args.images:
        pixel_values, pil_image, (oh, ow) = preprocess_image(img_path, processor, args.device)
        tokens = extract_patch_tokens(vision_model, pixel_values, args.layer)
        pca_tokens = compute_pca_rgb(tokens)
        
        # Reshape PCA to image
        N = tokens.shape[0]
        grid_size = int(math.isqrt(N))
        pca_grid = pca_tokens.reshape(grid_size, grid_size, 3)
        pca_image = Image.fromarray((pca_grid * 255).astype(np.uint8)).resize((ow, oh), Image.BILINEAR)
        
        # Plotting
        n_texts = len(args.texts)
        n_cols = 2 + n_texts
        fig, axes = plt.subplots(1, n_cols, figsize=(5 * n_cols, 5))
        if n_cols == 1: axes = [axes]
        
        axes[0].imshow(pil_image)
        axes[0].set_title("Original")
        axes[0].axis("off")
        
        axes[1].imshow(pca_image)
        axes[1].set_title("Vision PCA (RGB)")
        axes[1].axis("off")
        
        if need_text:
            heatmaps = compute_activation_heatmaps(model, tokenizer, tokens, args.texts, args.device)
            for i, (query, heatmap) in enumerate(zip(args.texts, heatmaps)):
                overlay = heatmap_to_image(heatmap, oh, ow, pil_image)
                axes[2+i].imshow(overlay)
                axes[2+i].set_title(f'Activation: "{query}"')
                axes[2+i].axis("off")
        
        plt.tight_layout()
        if args.output:
            plt.savefig(args.output)
        else:
            plt.show()
            
    if need_text:
        # Separate plot for Text PCA
        plt.figure(figsize=(10, 8))
        all_tokens = np.concatenate([tf["tokens"] for tf in text_features])
        all_words = [w for tf in text_features for w in tf["words"]]
        
        pca_t = compute_pca_rgb(all_tokens)
        plt.scatter(pca_t[:, 0], pca_t[:, 1], c=pca_t[:, 2], cmap='viridis', alpha=0.6)
        for i, word in enumerate(all_words):
            if i % 5 == 0: # Downsample labels to avoid clutter
                plt.text(pca_t[i, 0], pca_t[i, 1], word, fontsize=8)
        plt.title("Text Token PCA")
        plt.axis("off")
        if args.output:
            plt.savefig(args.output + "_text_pca.png")
        else:
            plt.show()

if __name__ == "__main__":
    import tyro
    main(tyro.cli(Args))
