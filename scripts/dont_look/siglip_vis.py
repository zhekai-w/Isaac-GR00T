import argparse
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from sklearn.decomposition import PCA
from transformers import AutoModel, AutoProcessor, SiglipVisionModel, SiglipImageProcessor

def normalize_rgb(data):
    """Normalize data to [0, 1] range for RGB visualization."""
    data = data.reshape(-1, data.shape[-1])
    d_min = data.min(axis=0)
    d_max = data.max(axis=0)
    denom = d_max - d_min
    denom[denom == 0] = 1.0
    normalized = (data - d_min) / denom
    return normalized

def apply_pca_rgb(features):
    """Reduce features to 3D using PCA and normalize to [0, 1]."""
    if features.ndim == 3:
        features = features[0]
    
    features_np = features.detach().cpu().numpy().astype(np.float32)
    
    pca = PCA(n_components=3)
    reduced = pca.fit_transform(features_np)
    
    return normalize_rgb(reduced)

def visualize_tokens(img_path, model, processor, use_pixel_shuffle=None, downsample_ratio=0.5, output_dir="vis_outputs"):
    os.makedirs(output_dir, exist_ok=True)
    
    image = Image.open(img_path).convert("RGB")
    inputs = processor(images=image, return_tensors="pt").to(model.device)
    pixel_values = inputs["pixel_values"]
    
    # 1. Raw SigLIP Output
    with torch.no_grad():
        if hasattr(model, "vision_model"):
            raw_output = model.vision_model(pixel_values=pixel_values, output_hidden_states=False, return_dict=True)
            raw_features = raw_output.last_hidden_state 
        else:
            raw_features = model(pixel_values=pixel_values).last_hidden_state
    
    # 2. Pixel Shuffle
    shuffled_features = None
    # If use_pixel_shuffle is not explicitly provided, check model attribute
    actual_shuffle = use_pixel_shuffle if use_pixel_shuffle is not None else getattr(model, "use_pixel_shuffle", False)
    
    if actual_shuffle:
        with torch.no_grad():
            vit_embeds = raw_features
            h = w = int(vit_embeds.shape[1] ** 0.5)
            # Handle cases where tokens are not a perfect square
            if h * w != vit_embeds.shape[1]:
                print(f"Warning: {vit_embeds.shape[1]} tokens not a square. Skipping shuffle visualization.")
                shuffled_features = None
            else:
                vit_embeds = vit_embeds.reshape(vit_embeds.shape[0], h, w, -1)
                scale_factor = downsample_ratio
                n, w_in, h_in, c_in = vit_embeds.size()
                x = vit_embeds.view(n, w_in, int(h_in * scale_factor), int(c_in / scale_factor))
                x = x.permute(0, 2, 1, 3).contiguous()
                x = x.view(
                    n, int(h_in * scale_factor), int(w_in * scale_factor), int(c_in / (scale_factor * scale_factor))
                )
                x = x.permute(0, 2, 1, 3).contiguous()
                shuffled_features = x.reshape(n, -1, x.shape[-1])
    
    # 3. Post-MLP
    with torch.no_grad():
        if hasattr(model, "mlp1"):
            mlp_input = shuffled_features if shuffled_features is not None else raw_features
            mlp_features = model.mlp1(mlp_input)
        elif hasattr(model, "dummy_mlp"):
            mlp_input = shuffled_features if shuffled_features is not None else raw_features
            mlp_features = model.dummy_mlp(mlp_input)
        else:
            mlp_features = None
        
    stages = {
        "raw": (raw_features, "Raw SigLIP"),
        "shuffle": (shuffled_features, "Pixel Shuffled"),
        "mlp": (mlp_features, "Post-MLP")
    }
    
    valid_stages = {k: v for k, v in stages.items() if v[0] is not None}
    fig, axes = plt.subplots(1, len(valid_stages) + 1, figsize=(5 * (len(valid_stages) + 1), 5))
    
    axes[0].imshow(image)
    axes[0].set_title("Original Image")
    axes[0].axis("off")
    
    for idx, (key, (features, title)) in enumerate(valid_stages.items(), 1):
        rgb_tokens = apply_pca_rgb(features)
        num_tokens = rgb_tokens.shape[0]
        grid_size = int(num_tokens**0.5)
        
        if grid_size * grid_size == num_tokens:
            h, w = grid_size, grid_size
        else:
            # Try to find a rectangle
            found = False
            for i in range(grid_size, 0, -1):
                if num_tokens % i == 0:
                    h, w = i, num_tokens // i
                    found = True
                    break
            if not found:
                rgb_tokens = np.pad(rgb_tokens, ((0, grid_size**2 - num_tokens), (0,0)), mode='constant')
                h, w = grid_size, grid_size
            
        rgb_grid = rgb_tokens.reshape(h, w, 3)
        axes[idx].imshow(rgb_grid, interpolation='nearest')
        axes[idx].set_title(title)
        axes[idx].axis("off")

    plt.tight_layout()
    save_path = os.path.join(output_dir, "siglip_pca_comparison.png")
    plt.savefig(save_path)
    print(f"Visualization saved to {save_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", type=str, required=True, help="Path to input image")
    parser.add_argument("--model_path", type=str, default=None, help="Path to Eagle model")
    args = parser.parse_args()
    
    import gr00t
    model_path = args.model_path or os.path.join(os.path.dirname(gr00t.__file__), "model", "backbone", "eagle2_hg_model")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    try:
        print(f"Attempting to load full model from {model_path}...")
        model = AutoModel.from_pretrained(model_path, trust_remote_code=True).to(device)
        processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
    except Exception as e:
        print(f"Could not load full model: {e}")
        print("Falling back to standalone SigLIP for visualization purposes...")
        # Standard SigLIP model for visualization
        siglip_id = "google/siglip-so400m-patch14-224"
        model = SiglipVisionModel.from_pretrained(siglip_id).to(device)
        processor = SiglipImageProcessor.from_pretrained(siglip_id)
        # Add dummy attributes to mimic Eagle model for the visualization script
        model.use_pixel_shuffle = True # Force true to visualize the process
        model.downsample_ratio = 0.5
        # MLP: 1152 (SigLIP) -> 2048 (LLM)
        model.dummy_mlp = torch.nn.Linear(1152, 2048).to(device)
        # Map dummy_mlp to mlp1 for the function
        model.mlp1 = model.dummy_mlp

    visualize_tokens(args.image, model, processor)

