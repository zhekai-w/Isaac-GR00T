import argparse
import torch
import math
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import os
import sys

# Add gr00t codebase to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from gr00t.model.transforms import build_eagle_processor
from gr00t.model.backbone.eagle_backbone import DEFAULT_EAGLE_PATH
from gr00t.model.backbone.eagle2_hg_model.modeling_eagle2_5_vl import Eagle2_5_VLForConditionalGeneration

def process_single_image(args, image_path, processor, model, device):
    print(f"Processing image: {image_path}")
    try:
        image = Image.open(image_path).convert("RGB")
    except Exception as e:
        print(f"Failed to open image at {image_path}: {e}")
        return

    conversation = [
        {
            "role": "user",
            "content": [{"type": "image", "image": image}, {"type": "text", "text": args.text_query}]
        }
    ]

    text_list = [processor.apply_chat_template(conversation, tokenize=False, add_generation_prompt=True)]
    image_inputs, video_inputs = processor.process_vision_info(conversation)
    
    inputs = processor(
        text=text_list, images=image_inputs, return_tensors="pt", padding=True
    ).to(device, dtype=torch.bfloat16)

    if "pixel_values" in inputs:
        inputs["pixel_values"] = inputs["pixel_values"].to(torch.bfloat16)

    if "image_sizes" in inputs:
        del inputs["image_sizes"]

    print("Running forward pass to extract attentions...")
    with torch.no_grad():
        outputs = model(
            **inputs,
            output_attentions=True,
            return_dict=True
        )

    attentions = outputs.attentions
    if not attentions:
        print("Model did not return attentions.")
        return
        
    num_total_layers = len(attentions)
    start_layer = max(0, num_total_layers - args.layers)
    
    input_ids = inputs["input_ids"][0]
    IMAGE_TOKEN_ID = model.config.image_token_index
    image_positions = torch.where(input_ids == IMAGE_TOKEN_ID)[0]
    
    if len(image_positions) == 0:
        print("No image tokens found in input_ids!")
        return
        
    num_img_tokens = len(image_positions)
    num_tokens_per_tile = getattr(model, "num_image_token", 256)
    grid_dim = int(math.sqrt(num_tokens_per_tile))
    
    print(f"Total image tokens: {num_img_tokens}, Tokens per tile: {num_tokens_per_tile}")
    if num_img_tokens > num_tokens_per_tile:
        print("Dynamic tiling active: visualizing the global thumbnail tile overlay.")

    input_tokens = processor.tokenizer.convert_ids_to_tokens(input_ids)
    
    query_idx = -1
    if args.track_token_type == "generation":
        query_idx = len(input_ids) - 1
    else:
        # Find the LAST occurrence of the text query, as causal LMs accumulate context rightward
        for i, token in enumerate(input_tokens):
            decoded_token = processor.tokenizer.decode([input_ids[i]])
            if args.text_query.lower() in decoded_token.lower() or args.text_query.lower() in token.lower():
                query_idx = i
                
    if query_idx == -1:
        print(f"Could not find '{args.text_query}'. Defaulting to last text token: '{processor.tokenizer.decode([input_ids[-1]])}'")
        query_idx = len(input_ids) - 1
    else:
        print(f"Tracking attention from token index {query_idx} ('{processor.tokenizer.decode([input_ids[query_idx]])}')")

    num_visualized_layers = num_total_layers - start_layer
    num_heads = attentions[0].shape[1]
    
    fig, axes = plt.subplots(num_visualized_layers, num_heads, figsize=(num_heads * 1.5, num_visualized_layers * 1.5))
    
    for l_idx in range(start_layer, num_total_layers):
        attn_layer = attentions[l_idx][0] 
        row_idx = l_idx - start_layer
        
        for h_idx in range(num_heads):
            attn_scores = attn_layer[h_idx, query_idx, image_positions].float().cpu().numpy()
            
            # The last 'num_tokens_per_tile' corresponds to the global thumbnail (or the full image if 1 tile)
            thumbnail_scores = attn_scores[-num_tokens_per_tile:]
            heatmap_data = thumbnail_scores.reshape(grid_dim, grid_dim)

            if num_visualized_layers == 1 and num_heads == 1:
                ax = axes
            elif num_visualized_layers == 1:
                ax = axes[h_idx]
            elif num_heads == 1:
                ax = axes[row_idx]
            else:
                ax = axes[row_idx, h_idx]
            
            # Plot the original image underneath so we have an exact spatial reference
            ax.imshow(image, extent=[0, grid_dim, grid_dim, 0])
            im = ax.imshow(heatmap_data, cmap='jet', interpolation='bilinear', alpha=0.5, extent=[0, grid_dim, grid_dim, 0])
            ax.axis('off')
            
            if row_idx == 0:
                ax.set_title(f"H{h_idx}", fontsize=8)
            if h_idx == 0:
                ax.text(-0.2, 0.5, f"L{l_idx+1}", va='center', ha='right', transform=ax.transAxes, fontsize=10, fontweight='bold')
                
    plt.tight_layout()
    
    # Save the output image
    base_name = os.path.splitext(os.path.basename(image_path))[0]
    out_file = os.path.join(args.output_dir, f"attention_{base_name}_{args.text_query}.png")
    plt.savefig(out_file, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {out_file}\n")


def main():
    parser = argparse.ArgumentParser(description="Visualize Text-to-Image Attention in EAGLE")
    parser.add_argument("--model_path", type=str, default="nvidia/GR00T-N1.5-3B", help="Path to fine-tuned model or default eagle path")
    parser.add_argument("--image_path", type=str, default="scripts/frame_000000.png", help="Path to single input image or a directory of images")
    parser.add_argument("--text_query", type=str, required=True, help="Text to append to the conversation prompt (e.g. 'red')")
    parser.add_argument("--track_token_type", type=str, choices=["query", "generation"], default="generation", help="Whether to track the attention of the exact text_query word or the final generation token.")
    parser.add_argument("--output_dir", type=str, default=".", help="Output directory for saved figures")
    parser.add_argument("--layers", type=int, default=4, help="Number of final layers to visualize")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"Loading processor from {DEFAULT_EAGLE_PATH}...")
    processor = build_eagle_processor(DEFAULT_EAGLE_PATH)

    print(f"Loading model from {args.model_path}...")
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
            args.model_path,
            torch_dtype=torch.bfloat16,
        )
        model = gr00t_model.backbone.eagle_model.to(device)
    except Exception as e:
        print(f"Could not load as GR00T_N1_5 checkpoint ({e}), falling back to Eagle2_5_VLForConditionalGeneration...")
        model = Eagle2_5_VLForConditionalGeneration.from_pretrained(
            args.model_path,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
        ).to(device)
    finally:
        transformers.AutoConfig.from_pretrained = orig_from_pretrained
        
    model.eval()

    if os.path.isdir(args.image_path):
        import glob
        image_files = sorted(glob.glob(os.path.join(args.image_path, "*.[pP][nN][gG]")) + glob.glob(os.path.join(args.image_path, "*.[jJ][pP][gG]")))
        if not image_files:
            print(f"No image files found in directory {args.image_path}")
            return
        print(f"Found {len(image_files)} image(s) to process.")
        for img_file in image_files:
            process_single_image(args, img_file, processor, model, device)
    else:
        process_single_image(args, args.image_path, processor, model, device)

if __name__ == "__main__":
    main()
