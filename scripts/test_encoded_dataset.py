# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Verify that a video-encoded LeRobot dataset (produced by encode_videos.py) is
compatible with the GR00T training pipeline.

Checks:
  1. Dataset loads without errors (modality.json + info.json are valid)
  2. All video files exist on disk
  3. Random-access frame decoding returns correct shapes, dtype, and value range
  4. torch DataLoader runs N batches without error (simulates training loop)
  5. Timing report per backend

Usage:
    python test_encoded_dataset.py --dataset_path /path/to/dataset_encoded
    python test_encoded_dataset.py --dataset_path /path/to/dataset_encoded --backends torchcodec pyav --num_workers 0 4 --batches 20
"""

import json
import pathlib
import time
from dataclasses import dataclass, field
from typing import List, Literal

import numpy as np
import torch
import tyro
from torch.utils.data import DataLoader

from gr00t.data.dataset import (
    LE_ROBOT_MODALITY_FILENAME,
    LeRobotSingleDataset,
    ModalityConfig,
)
from gr00t.data.embodiment_tags import EMBODIMENT_TAG_MAPPING, EmbodimentTag


# ─────────────────────────────────────────────────────────────────────────────
# CLI config
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class ArgsConfig:
    dataset_path: str
    """Path to the encoded dataset directory."""

    embodiment_tag: Literal[tuple(EMBODIMENT_TAG_MAPPING.keys())] = "new_embodiment"
    """Embodiment tag to use."""

    backends: List[str] = field(default_factory=lambda: ["torchcodec", "pyav"])
    """Video backends to test. Options: torchcodec, pyav, video_reader."""

    num_workers: List[int] = field(default_factory=lambda: [12])
    """DataLoader num_workers values to test."""

    prefetch_factor: int = 4
    """Prefetch factor for data loading."""

    pin_memory: bool = False
    """Whether to pin memory for data loading."""

    batches: int = 10
    """Number of batches to run per (backend, num_workers) combination."""

    batch_size: int = 32
    """Batch size for DataLoader test."""


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _pass(msg: str):
    print(f"  \033[92m[PASS]\033[0m {msg}")

def _fail(msg: str):
    print(f"  \033[91m[FAIL]\033[0m {msg}")

def _info(msg: str):
    print(f"  \033[94m[INFO]\033[0m {msg}")

def _section(title: str):
    print(f"\n{'─'*60}")
    print(f"  {title}")
    print(f"{'─'*60}")


def get_modality_keys(dataset_path: pathlib.Path) -> dict[str, list[str]]:
    modality_path = dataset_path / LE_ROBOT_MODALITY_FILENAME
    with open(modality_path, "r") as f:
        meta = json.load(f)
    result = {}
    for key, items in meta.items():
        # "annotation" is internal metadata used by the language modality path,
        # not a direct modality that can be passed to LeRobotSingleDataset.
        if key == "annotation":
            continue
        result[key] = [f"{key}.{item}" for item in items]
    return result


def build_modality_configs(modality_keys: dict[str, list[str]]) -> dict[str, ModalityConfig]:
    configs = {}
    for modality_type, keys in modality_keys.items():
        if not keys:
            continue
        configs[modality_type] = ModalityConfig(
            delta_indices=[0],
            modality_keys=keys,
        )
    return configs


# ─────────────────────────────────────────────────────────────────────────────
# Test 1 — dataset structure
# ─────────────────────────────────────────────────────────────────────────────

def check_dataset_structure(dataset_path: pathlib.Path):
    _section("Test 1: Dataset structure")
    errors = 0

    # info.json
    info_path = dataset_path / "meta" / "info.json"
    if not info_path.exists():
        _fail(f"meta/info.json missing")
        errors += 1
    else:
        with open(info_path) as f:
            info = json.load(f)
        _pass(f"meta/info.json found  (fps={info['fps']}, episodes={info['total_episodes']})")

        video_path_tmpl = info.get("video_path")
        if not video_path_tmpl:
            _fail("info.json has no 'video_path' — dataset may not have been encoded")
            errors += 1
        else:
            _pass(f"video_path template: {video_path_tmpl}")

        video_features = {k: v for k, v in info["features"].items() if v["dtype"] == "video"}
        image_features = {k: v for k, v in info["features"].items() if v["dtype"] == "image"}
        _pass(f"video features : {list(video_features.keys())}")
        if image_features:
            _fail(f"image features still present (not encoded): {list(image_features.keys())}")
            errors += 1

    # modality.json
    modality_path = dataset_path / LE_ROBOT_MODALITY_FILENAME
    if not modality_path.exists():
        _fail(f"{LE_ROBOT_MODALITY_FILENAME} missing — required by LeRobotSingleDataset")
        errors += 1
    else:
        _pass(f"{LE_ROBOT_MODALITY_FILENAME} found")

    return errors == 0, info if errors == 0 else None


# ─────────────────────────────────────────────────────────────────────────────
# Test 2 — video files on disk
# ─────────────────────────────────────────────────────────────────────────────

def check_video_files(dataset_path: pathlib.Path, info: dict):
    _section("Test 2: Video files on disk")
    errors = 0
    tmpl = info["video_path"]
    total_eps = info["total_episodes"]
    chunks_size = info["chunks_size"]
    video_keys = [k for k, v in info["features"].items() if v["dtype"] == "video"]

    for ep_idx in range(total_eps):
        chunk = ep_idx // chunks_size
        for vk in video_keys:
            rel = tmpl.format(episode_chunk=chunk, video_key=vk, episode_index=ep_idx)
            full = dataset_path / rel
            if not full.exists():
                _fail(f"Missing: {rel}")
                errors += 1

    if errors == 0:
        total = total_eps * len(video_keys)
        _pass(f"All {total} video files present ({total_eps} episodes × {len(video_keys)} cameras)")
    return errors == 0


# ─────────────────────────────────────────────────────────────────────────────
# Test 3 — single-item decode
# ─────────────────────────────────────────────────────────────────────────────

def check_single_item_decode(
    dataset_path: pathlib.Path,
    info: dict,
    backend: str,
    embodiment_tag: EmbodimentTag,
):
    _section(f"Test 3: Single-item decode  [backend={backend}]")

    modality_keys = get_modality_keys(dataset_path)
    # strip dummy tensors
    if "state" in modality_keys:
        modality_keys["state"] = [k for k in modality_keys["state"] if "dummy" not in k]

    modality_configs = build_modality_configs(modality_keys)
    video_keys = modality_keys.get("video", [])

    try:
        dataset = LeRobotSingleDataset(
            dataset_path=str(dataset_path),
            modality_configs=modality_configs,
            embodiment_tag=embodiment_tag,
            video_backend=backend,
        )
    except Exception as e:
        _fail(f"LeRobotSingleDataset init failed: {e}")
        return False, None

    _pass(f"Dataset loaded  ({len(dataset)} items)")

    # sample a few indices spread across the dataset
    indices = [0, len(dataset) // 4, len(dataset) // 2, len(dataset) - 1]
    errors = 0

    for idx in indices:
        try:
            item = dataset[idx]
        except Exception as e:
            _fail(f"dataset[{idx}] raised: {e}")
            errors += 1
            continue

        for vk in video_keys:
            if vk not in item:
                _fail(f"dataset[{idx}]: key '{vk}' missing from item")
                errors += 1
                continue

            arr = item[vk]
            if not isinstance(arr, np.ndarray):
                _fail(f"dataset[{idx}]['{vk}']: expected np.ndarray, got {type(arr)}")
                errors += 1
                continue

            # shape: (delta, H, W, C) or (H, W, C)
            if arr.ndim not in (3, 4):
                _fail(f"dataset[{idx}]['{vk}']: unexpected ndim={arr.ndim}, shape={arr.shape}")
                errors += 1
                continue

            vmin, vmax = arr.min(), arr.max()
            # Raw dataset output is [0, 255] (uint8-range float); transforms
            # applied during training normalize to [0, 1]. Accept both.
            if vmin < -0.01 or vmax > 255.01:
                _fail(f"dataset[{idx}]['{vk}']: values out of expected range — min={vmin:.3f} max={vmax:.3f}")
                errors += 1
                continue

        if errors == 0:
            vk0 = video_keys[0] if video_keys else None
            shape_str = str(item[vk0].shape) if vk0 else "N/A"
            range_str = f"[{item[vk0].min():.0f}, {item[vk0].max():.0f}]" if vk0 else ""
            _pass(f"dataset[{idx}]: OK — shape {shape_str}  range {range_str}")

    return errors == 0, dataset


# ─────────────────────────────────────────────────────────────────────────────
# Test 4 — DataLoader (simulates training loop)
# ─────────────────────────────────────────────────────────────────────────────

def check_dataloader(dataset, backend: str, num_workers: int, batch_size: int, batches: int, prefetch_factor: int = 4, pin_memory: bool = False):
    _section(f"Test 4: DataLoader  [backend={backend}, num_workers={num_workers}, batch_size={batch_size}]")

    def collate_fn(items):
        batch = {}
        for key in items[0]:
            try:
                batch[key] = np.stack([item[key] for item in items])
            except Exception:
                batch[key] = [item[key] for item in items]
        return batch

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=True,
        collate_fn=collate_fn,
        pin_memory=pin_memory,
        prefetch_factor=prefetch_factor if num_workers > 0 else None,
        persistent_workers=(num_workers > 0),
    )

    errors = 0
    t0 = time.perf_counter()

    try:
        for i, batch in enumerate(loader):
            if i >= batches:
                break
            if i == 0:
                # verify batch shapes on first iteration
                for key, val in batch.items():
                    if isinstance(val, np.ndarray):
                        _info(f"batch['{key}'].shape = {val.shape}")
    except Exception as e:
        _fail(f"DataLoader failed at batch {i}: {e}")
        errors += 1

    elapsed = time.perf_counter() - t0
    actual_batches = min(batches, len(loader))

    if errors == 0:
        _pass(
            f"{actual_batches} batches in {elapsed:.2f}s  "
            f"({elapsed/actual_batches*1000:.1f} ms/batch)"
        )

    return errors == 0, elapsed


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main(config: ArgsConfig):
    dataset_path = pathlib.Path(config.dataset_path).resolve()
    embodiment_tag = EmbodimentTag(config.embodiment_tag)

    print(f"\n{'='*60}")
    print(f"  Dataset : {dataset_path}")
    print(f"  Backends: {config.backends}")
    print(f"  Workers : {config.num_workers}")
    print(f"{'='*60}")

    # ── structure check (once) ────────────────────────────────────────────────
    ok, info = check_dataset_structure(dataset_path)
    if not ok:
        print("\n\033[91mStructure check failed. Fix errors above before continuing.\033[0m")
        return

    ok = check_video_files(dataset_path, info)
    if not ok:
        print("\n\033[91mMissing video files. Run encode_videos.py first.\033[0m")
        return

    # ── per-backend tests ─────────────────────────────────────────────────────
    results = []

    for backend in config.backends:
        decode_ok, dataset = check_single_item_decode(dataset_path, info, backend, embodiment_tag)
        if not decode_ok or dataset is None:
            results.append((backend, None, False, None))
            continue

        for nw in config.num_workers:
            loader_ok, elapsed = check_dataloader(
                dataset, backend, nw, config.batch_size, config.batches,
                prefetch_factor=config.prefetch_factor,
                pin_memory=config.pin_memory,
            )
            results.append((backend, nw, loader_ok, elapsed))

    # ── summary ───────────────────────────────────────────────────────────────
    _section("Summary")
    all_passed = True
    for backend, nw, ok, elapsed in results:
        if nw is None:
            status = "\033[91mFAIL\033[0m (decode error)"
            all_passed = False
        elif ok:
            ms = elapsed / config.batches * 1000
            status = f"\033[92mPASS\033[0m  {ms:.1f} ms/batch"
        else:
            status = "\033[91mFAIL\033[0m"
            all_passed = False
        workers_str = f"workers={nw}" if nw is not None else "decode"
        print(f"  {backend:<16} {workers_str:<12} {status}")

    print()
    if all_passed:
        print("\033[92mAll checks passed — dataset is ready for training.\033[0m\n")
    else:
        print("\033[91mSome checks failed — see details above.\033[0m\n")


if __name__ == "__main__":
    config = tyro.cli(ArgsConfig)
    main(config)
