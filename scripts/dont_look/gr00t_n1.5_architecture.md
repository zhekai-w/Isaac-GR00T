# GR00T-N1.5 Architecture Notes

Summary of architecture details from exploring the codebase for text-to-image attention visualization.

## Top-level: Dual-Brain Architecture

```
Input (images + text + state) → Backbone → Action Head → Predicted actions
```

- **Class**: `GR00T_N1_5` (`gr00t/model/gr00t_n1.py`)
- **Components**:
  - `self.backbone = EagleBackbone(...)` — vision-language understanding
  - `self.action_head = FlowmatchingActionHead(...)` — diffusion-based action generation

---

## 1. Vision-Language Backbone (Eagle)

### EagleBackbone
- File: `gr00t/model/backbone/eagle_backbone.py`
- Wraps `Eagle2_5_VLForConditionalGeneration` (the actual VL model)
- Projects LLM hidden size (2048) → shared dim (1536) via `eagle_linear`
- LLM layers truncated to `select_layer` (default keeps only 12 of full Qwen2)
- Output: `(B, S, 1536)` features + attention mask

### Eagle2_5_VL Model
- File: `gr00t/model/backbone/eagle2_hg_model/modeling_eagle2_5_vl.py`
- Has:
  - `self.vision_model` — SiglipVisionModel or RADIOModel
  - `self.language_model` — Qwen2ForCausalLM (in this build)
  - `self.mlp1` — projects vision features into language embedding space
- Supports `output_attentions=True` to expose per-layer attention tensors

### LLM config (this checkpoint)
| Property | Value |
|---|---|
| LLM layers | 12 (truncated) |
| Attention heads | 16 |
| KV heads | 8 (GQA) |
| Hidden size | 2048 |
| Attention impl | `eager` required to return weights |

---

## 2. Vision Tokenization (Image Patches)

- Input image resized to 448×448 (one tile)
- Patch size = 28 → 16×16 = 256 patch tokens per tile
- Pixel shuffle (if enabled) keeps count at 256
- Each patch = one image token = one image **key** in LLM self-attention

### Dynamic tiling (high-res images)
- Input image split into multiple 448×448 tiles + 1 global thumbnail
- Example from script log: `2304 image tokens = 9 tiles × 256 patches`
  - 8 high-res tiles + 1 global thumbnail tile
- Each tile gets its own 256 tokens; all concatenated into LLM sequence
- Visualization script currently uses the thumbnail tile only

### Terminology map
| Concept | Same thing |
|---|---|
| Image patch | 16×16 pixel region |
| Image token | ViT output for that patch |
| Image key (K) | That token in LLM attention |
| Image value (V) | Same token, different projection |

---

## 3. Attention Flow (for visualization)

Inside the LLM (causal self-attention):
```
Q = text_token_hidden  @ W_q
K = image_token_hidden @ W_k
V = image_token_hidden @ W_v
attn = softmax(Q · K^T / √d) · V
```

Shape of attention tensor per layer: `(batch, heads, seq_len, seq_len)`.
Extracting the text-to-image slice: `attn[:, :, text_token_idx, image_positions]` → `(heads, 256)` for a single tile.

---

## 4. Action Head (DiT with Cross-Attention)

- File: `gr00t/model/action_head/flow_matching_action_head.py`
- Uses `DiT` from `cross_attention_dit.py`
- Flow:
  1. VL features get LayerNorm + self-attention (`vl_self_attention`)
  2. Concatenate `[state, future_tokens, noisy_actions]` → action embeddings `(B, T, inner_dim)`
  3. DiT transformer blocks: action tokens (Q) cross-attend to VL features (K, V)
  4. Output projected to action velocity

### DiT config
- 12 transformer blocks
- 8 attention heads × 64 dim = 512 inner dim (per config)
- `cross_attention_dim` = 1536 (matches projected VL feature dim)
- Optional `interleave_self_attention`: alternate self/cross-attn layers
- Timestep embedding drives Ada-LayerNorm for diffusion conditioning

Parameter counts (observed):
- DiT: ~550M params
- SelfAttentionTransformer (VL processor): ~201M params

---

## 5. Concepts Relevant to Attention Visualization

### CLS-like aggregators / attention sinks
In ViT: `[CLS]` = extra token that aggregates global info. Eagle has no explicit CLS, but LLMs learn to use certain patches (often first patch, register-like tokens) as attention sinks where all other tokens dump their attention. Object tokens like "cup" often attend to these sinks instead of the object pixels.

Reference: "Vision Transformers Need Registers" (Darcet et al., 2023).

### BPE leakage
BPE = subword tokenizer. In causal LM, later tokens already see prior context through attention.
- Prompt `"the red cup"` → tokens `["the", " red", " cup"]`
- By the time the model processes `" cup"`, its hidden state already carries `"red"` info
- So the `" cup"` token's attention may fire on red regions (inherited context) rather than cup-specific shape
- The preceding `" the"` often looks most object-focused because the model preloads the upcoming noun

### Why color attention looks sharper than object attention
1. Color = low-level patch feature, object = compositional (multi-patch, shape, parts)
2. Causal LM diffuses attention once context is already established
3. Thumbnail tile (256 patches) is coarse; small objects wash out
4. Attention sinks capture object tokens more than color tokens

---

## 6. Files Touched in This Session

| File | Purpose |
|---|---|
| `scripts/visualize_text_image_cross_attention.py` | New — per-text-token spatial attention maps from Eagle LLM |
| `scripts/visualize_eagle_attention.py` | Existing — single query token attention, per layer/head |

### Script behavior
- Patches the transformer config with `_attn_implementation = "eager"` to enable attention weights output
- Loads `GR00T_N1_5` then pulls out `backbone.eagle_model` for inference
- Filters to only the user's text query tokens (excludes system/template tokens)
- Averages attention over last N layers (default 4) and all heads
- Raw 16×16 heatmap with `interpolation="nearest"` (no smoothing)
- Output naming: `cross_attn_<image>_<query>.png`
