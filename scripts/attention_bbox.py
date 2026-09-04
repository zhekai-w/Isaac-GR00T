"""
Bounding-box grounding utilities for attention visualization scripts.

Turns "does the model look at the object?" from an eyeball judgement into a
number: an open-vocabulary detector (OWLv2 / Grounding DINO) produces a box for
the queried object, the box is projected onto the ViT patch grid, and the
attention mass falling inside it is scored.

Main metric is the *concentration ratio*:

    concentration = (attention mass inside box) / (box area / image area)

which is 1.0 for a uniform (uninformative) attention map and >1 when attention
concentrates on the object.  Unlike a plain mean-inside-the-box, it stays
comparable across boxes of different size.

Geometry note: the Eagle2.5 processor builds its global thumbnail tile with a
plain resize of the full image to 224x224 (no letterbox, no pad, no crop --
see image_processing_eagle2_5_vl_fast.py:326-329, pad_during_tiling=false), so
the pixel -> patch mapping is a pure linear scale with no offset.
"""

import numpy as np
import torch
from matplotlib.patches import Rectangle

OWLV2_DEFAULT = "google/owlv2-base-patch16-ensemble"
OWLVIT_DEFAULT = "google/owlvit-base-patch32"
GDINO_DEFAULT = "IDEA-Research/grounding-dino-tiny"
OMDET_DEFAULT = "omlab/omdet-turbo-swin-tiny-hf"

DEFAULT_MODEL_IDS = {
    "owlv2": OWLV2_DEFAULT,
    "owlvit": OWLVIT_DEFAULT,
    "gdino": GDINO_DEFAULT,
    "omdet": OMDET_DEFAULT,
}

# (source, model_id, device) -> (processor, model)
_DETECTOR_CACHE = {}


def default_model_id(source):
    return DEFAULT_MODEL_IDS[source]


def get_detector(source, model_id=None, device="cuda"):
    """Load (and cache) an open-vocabulary detector.  One load per process."""
    model_id = model_id or default_model_id(source)
    key = (source, model_id, device)
    if key in _DETECTOR_CACHE:
        return _DETECTOR_CACHE[key]

    if source == "owlv2":
        from transformers import Owlv2ForObjectDetection, Owlv2Processor

        processor = Owlv2Processor.from_pretrained(model_id)
        model = Owlv2ForObjectDetection.from_pretrained(model_id).to(device).eval()
    elif source == "owlvit":
        from transformers import OwlViTForObjectDetection, OwlViTProcessor

        processor = OwlViTProcessor.from_pretrained(model_id)
        model = OwlViTForObjectDetection.from_pretrained(model_id).to(device).eval()
    elif source == "omdet":
        from transformers import AutoProcessor, OmDetTurboForObjectDetection

        processor = AutoProcessor.from_pretrained(model_id)
        model = OmDetTurboForObjectDetection.from_pretrained(model_id).to(device).eval()
    elif source == "gdino":
        from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor

        processor = AutoProcessor.from_pretrained(model_id)
        model = (
            AutoModelForZeroShotObjectDetection.from_pretrained(model_id).to(device).eval()
        )
    else:
        raise ValueError(f"Unknown box source: {source}")

    _DETECTOR_CACHE[key] = (processor, model)
    return processor, model


def _post_process(processor, outputs, threshold, target_sizes, input_ids=None):
    """Call whichever post-processing entry point this transformers version has."""
    kwargs = dict(threshold=threshold, target_sizes=target_sizes)
    if input_ids is not None:
        kwargs["input_ids"] = input_ids
    if hasattr(processor, "post_process_grounded_object_detection"):
        try:
            return processor.post_process_grounded_object_detection(outputs, **kwargs)
        except TypeError:
            kwargs.pop("input_ids", None)
            return processor.post_process_grounded_object_detection(outputs, **kwargs)
    kwargs.pop("input_ids", None)
    return processor.post_process_object_detection(outputs, **kwargs)


def _label_matches(results, target_idx, target_text):
    """Boolean mask over detections whose label is the target query.

    Different models report labels differently: OWLv2/OWL-ViT give an integer
    index into the prompt list, while Grounding DINO gives the matched token
    span, which for a multi-phrase prompt can merge phrases into things like
    "mango pepper yellow pepper".  Exact string equality therefore throws away
    correct detections, so string labels are matched by word containment.
    """
    # transformers may include a "text_labels" key whose value is None, so treat
    # missing and None the same.
    labels = results.get("text_labels") or results.get("labels")
    if labels is None or len(labels) == 0:
        return None
    if torch.is_tensor(labels):
        return labels.detach().cpu() == target_idx
    target_words = set(target_text.lower().strip().rstrip(".").split())
    return torch.tensor(
        [target_words <= set(str(l).lower().strip().rstrip(".").split()) for l in labels],
        dtype=torch.bool,
    )


def _wins_its_box(boxes, scores, mask, group_iou=0.7):
    """Drop target detections that a competing label scores higher on.

    Grounding DINO and OmDet return a score for *every* (box, label) pair rather
    than one label per box, so a detection labelled with the target can coexist
    with a higher-scoring detection of the same object under a different label.
    Keeping only the boxes where the target label wins turns the negatives into
    a genuine argmax over labels.
    """
    from torchvision.ops import box_iou

    if mask.all():
        return mask
    ious = box_iou(boxes, boxes)
    keep = mask.clone()
    others = (~mask).nonzero(as_tuple=True)[0]
    for i in mask.nonzero(as_tuple=True)[0].tolist():
        overlapping = others[ious[i, others] >= group_iou]
        if overlapping.numel() and (scores[overlapping] > scores[i]).any():
            keep[i] = False
    return keep


def detect_boxes(image, query, source="owlv2", model_id=None, threshold=0.1,
                 device="cuda", nms_iou=0.5, max_boxes=0, negatives=(),
                 return_negatives=False):
    """Detect every instance of `query` in `image` (PIL RGB).

    Returns a list of ((x0,y0,x1,y1), score) sorted by score descending, empty
    if nothing passes `threshold`.  With `return_negatives`, returns
    `(target_list, {negative_phrase: list})` so the distractors can be plotted
    and scored too.  OWLv2 emits overlapping duplicates for the
    same object, so NMS is applied (`nms_iou`; set <=0 to disable).  `max_boxes`
    caps the count (0 = keep all).

    `negatives` are competing labels for the other objects in the scene, scored
    in the same forward pass; only detections the model assigns to `query` are
    kept.  These scores are open-vocabulary similarities, not calibrated
    probabilities -- a rare word like "mango" peaks around 0.15 where "apple"
    reaches 0.80 -- so with a single query a distractor object can outrank the
    real one simply because nothing else competes for it.  Naming the
    distractors fixes that far more reliably than lowering the threshold.
    """
    processor, model = get_detector(source, model_id, device)
    W, H = image.size
    negatives = [n for n in negatives if n.strip() and n.strip() != query.strip()]

    prompts = [query] + negatives

    if source in ("owlv2", "owlvit"):
        inputs = processor(text=[prompts], images=image, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = model(**inputs)
        if source == "owlv2":
            # OWLv2 pads the image to a square on the bottom/right before resizing.
            # Passing the *padded* square size back means the returned boxes land
            # in original-pixel coordinates directly; (H, W) silently skews them.
            side = max(H, W)
            target_sizes = torch.tensor([[side, side]], device=device)
        else:
            # OWL-ViT does not pad, so the true image size is correct here.
            target_sizes = torch.tensor([[H, W]], device=device)
        results = _post_process(processor, outputs, threshold, target_sizes)[0]
    elif source == "omdet":
        inputs = processor(images=image, text=prompts, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = model(**inputs)
        results = processor.post_process_grounded_object_detection(
            outputs, text_labels=prompts, threshold=threshold, nms_threshold=1.0,
            target_sizes=[(H, W)],
        )[0]
    else:
        # Grounding DINO wants lowercase phrases terminated with periods.
        text = ". ".join(p.lower().strip().rstrip(".") for p in prompts) + "."
        inputs = processor(images=image, text=text, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = model(**inputs)
        target_sizes = torch.tensor([[H, W]], device=device)
        results = _post_process(
            processor, outputs, threshold, target_sizes, input_ids=inputs["input_ids"]
        )[0]

    all_scores = results["scores"].detach().float().cpu()
    all_boxes = results["boxes"].detach().float().cpu()
    if all_scores.numel() == 0:
        return ([], {}) if return_negatives else []

    def select(prompt_idx, prompt_text, exclude=None):
        """Boxes this prompt wins, NMS'd, clipped and sorted by score.

        `exclude` masks out detections the target query already claimed.
        Grounding DINO merges phrase spans, so one detection can come back
        labelled "mango pepper yellow pepper" and match both "mango" and
        "yellow pepper" by word containment; without this the target's own box
        is re-emitted as a distractor under the other name.
        """
        boxes, scores = all_boxes, all_scores
        mask = None
        if len(prompts) > 1:
            mask = _label_matches(results, prompt_idx, prompt_text)
            if mask is not None:
                mask = _wins_its_box(boxes, scores, mask)
                if exclude is not None:
                    mask = mask & ~exclude
                boxes, scores = boxes[mask], scores[mask]
                if scores.numel() == 0:
                    return [], mask

        if nms_iou > 0 and scores.numel() > 1:
            from torchvision.ops import nms

            keep = nms(boxes, scores, nms_iou)
            boxes, scores = boxes[keep], scores[keep]

        out = []
        for i in torch.argsort(scores, descending=True).tolist():
            x0, y0, x1, y1 = boxes[i].tolist()
            x0, x1 = max(0.0, min(x0, W)), max(0.0, min(x1, W))
            y0, y1 = max(0.0, min(y0, H)), max(0.0, min(y1, H))
            if x1 <= x0 or y1 <= y0:
                continue
            out.append(((x0, y0, x1, y1), float(scores[i].item())))
            if max_boxes and len(out) >= max_boxes:
                break
        return out, mask

    target, target_mask = select(0, query)
    if not return_negatives:
        return target
    return target, {
        n: select(i + 1, n, exclude=target_mask)[0] for i, n in enumerate(negatives)
    }


def box_patch_weights(box, img_size, grid_dim):
    """Fractional overlap of `box` with each patch cell -> (grid_dim, grid_dim) in [0,1].

    Fractional rather than binary: at 16x16 a patch spans 40px of a 640px frame,
    so a small object covers ~2 patches and binary in/out would quantize the
    metric badly.  Index order is [row, col], matching the row-major patch order
    of the ViT.
    """
    W, H = img_size
    x0, y0, x1, y1 = box
    xs = np.linspace(0.0, float(W), grid_dim + 1)
    ys = np.linspace(0.0, float(H), grid_dim + 1)
    wx = np.clip(np.minimum(xs[1:], x1) - np.maximum(xs[:-1], x0), 0.0, None) / (W / grid_dim)
    wy = np.clip(np.minimum(ys[1:], y1) - np.maximum(ys[:-1], y0), 0.0, None) / (H / grid_dim)
    return np.outer(wy, wx)


def union_patch_weights(boxes, img_size, grid_dim):
    """Patch weights for a set of boxes, combined with elementwise max.

    Max rather than sum so overlapping boxes on the same object do not
    double-count area (which would inflate area_frac and deflate concentration).
    """
    w = np.zeros((grid_dim, grid_dim), dtype=np.float64)
    for box in boxes:
        w = np.maximum(w, box_patch_weights(box, img_size, grid_dim))
    return w


def score_attention(attn_vec, weights):
    """Score one attention row (num_patches,) against patch weights (grid, grid).

    The raw attention row is a slice of a softmax that also covers system and
    text tokens, so it is renormalized over the image patches first -- otherwise
    "mass inside the box" is not a fraction of anything meaningful.
    """
    a = np.asarray(attn_vec, dtype=np.float64).reshape(-1)
    w = np.asarray(weights, dtype=np.float64).reshape(-1)
    assert a.size == w.size, f"attention has {a.size} patches, weights have {w.size}"

    total = a.sum()
    p = a / total if total > 0 else np.zeros_like(a)
    mass_in = float((p * w).sum())
    area_frac = float(w.sum() / w.size)
    concentration = mass_in / area_frac if area_frac > 0 else float("nan")
    return {"mass_in": mass_in, "area_frac": area_frac, "concentration": concentration}


def draw_box(ax, box, color="lime", linewidth=1.5):
    """Draw a detected box on a matplotlib axis showing the original image."""
    x0, y0, x1, y1 = box
    ax.add_patch(
        Rectangle(
            (x0, y0), x1 - x0, y1 - y0,
            linewidth=linewidth, edgecolor=color, facecolor="none",
        )
    )


def policy_preprocess(image, detections=(), scale=0.95, size=224):
    """Mimic the deployed policy's view: center-crop to `scale`, resize to `size`.

    The policy pipeline is VideoCrop(scale=0.95) -> VideoResize(224, 224) with
    eval mode giving a center crop (data_config.py:172-173, policy.py:97), so a
    full-resolution frame fed straight to this script is *not* what the policy
    sees.  Detections are transformed alongside the image; any box that falls
    entirely outside the crop is dropped.
    """
    W, H = image.size
    crop_w, crop_h = int(W * scale), int(H * scale)
    left = int(round((W - crop_w) / 2.0))
    top = int(round((H - crop_h) / 2.0))

    cropped = image.crop((left, top, left + crop_w, top + crop_h)).resize(
        (size, size), resample=3  # BICUBIC, matching the Eagle processor
    )

    sx, sy = size / crop_w, size / crop_h

    def remap(dets):
        out = []
        for box, score in dets:
            x0, y0, x1, y1 = box
            new_box = (
                max(0.0, min((x0 - left) * sx, size)),
                max(0.0, min((y0 - top) * sy, size)),
                max(0.0, min((x1 - left) * sx, size)),
                max(0.0, min((y1 - top) * sy, size)),
            )
            if new_box[2] <= new_box[0] or new_box[3] <= new_box[1]:
                continue
            out.append((new_box, score))
        return out

    if isinstance(detections, dict):
        return cropped, {k: remap(v) for k, v in detections.items()}
    return cropped, remap(detections)
