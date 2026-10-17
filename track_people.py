#!/usr/bin/env python3
"""Detect people on frame zero and track them through HQ-OpenHumanVid clips."""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import re
import sys
import tarfile
import tempfile
from contextlib import nullcontext
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator


PART_COUNT = 12
CLIP_RE = re.compile(r"^part_(\d{3})/[0-9a-f]{2}/[0-9a-f]{2}/([0-9a-f]{32})\.mp4$")
_WORKER = None


@dataclass(frozen=True)
class Clip:
    part: int
    clip_id: str
    source: str
    video: bytes


def archive_path(dataset: Path, part: int) -> Path:
    return dataset / f"part_{part:03d}.tar.gz"


def output_path(output: Path, part: int, clip_id: str) -> Path:
    return output / f"part_{part:03d}" / clip_id


def iter_archive_clips(
    archive: Path, part: int, output: Path, stats: dict[str, int] | None = None
) -> Iterator[Clip]:
    """Read one gzip stream once; never unpack a whole part to disk."""
    with tarfile.open(archive, mode="r|gz") as tar:
        for member in tar:
            if not member.isfile() or not member.name.lower().endswith(".mp4"):
                continue
            match = CLIP_RE.fullmatch(member.name)
            if not match or int(match.group(1)) != part:
                raise ValueError(f"Unexpected MP4 path in {archive}: {member.name}")
            clip_id = match.group(2)
            if (output_path(output, part, clip_id) / "metadata.json").is_file():
                if stats is not None:
                    stats["skipped"] += 1
                continue
            source = tar.extractfile(member)
            if source is None:
                raise ValueError(f"Cannot read {member.name} from {archive}")
            video = source.read()
            if len(video) != member.size:
                raise ValueError(f"Truncated video in {archive}: {member.name}")
            yield Clip(part, clip_id, member.name, video)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True, help="Directory containing part_###.tar.gz")
    parser.add_argument("--start-part", type=int, required=True, help="First part, inclusive (1-12)")
    parser.add_argument("--end-part", type=int, required=True, help="Last part, inclusive (1-12)")
    parser.add_argument("--workers", type=int, required=True, help="Number of parallel GPU processes")
    parser.add_argument("--output", type=Path, default=Path("masks"), help="Mask output directory")
    parser.add_argument("--gpus", help="Comma-separated CUDA device indices; defaults to all visible GPUs")
    parser.add_argument("--grounding-model", default="IDEA-Research/grounding-dino-tiny")
    parser.add_argument("--sam2-model", default="facebook/sam2.1-hiera-large")
    parser.add_argument("--box-threshold", type=float, default=0.35)
    parser.add_argument("--text-threshold", type=float, default=0.25)
    parser.add_argument("--nms-threshold", type=float, default=0.5)
    parser.add_argument("--dry-run", action="store_true", help="Validate part selection without loading models")
    args = parser.parse_args(argv)
    if not 1 <= args.start_part <= args.end_part <= PART_COUNT:
        parser.error("Part range must satisfy 1 <= start-part <= end-part <= 12")
    if args.workers < 1:
        parser.error("--workers must be at least 1")
    for name in ("box_threshold", "text_threshold", "nms_threshold"):
        if not 0 < getattr(args, name) < 1:
            parser.error(f"--{name.replace('_', '-')} must be between 0 and 1")
    return args


def validate_archives(dataset: Path, parts: range) -> list[Path]:
    if not dataset.is_dir():
        raise FileNotFoundError(f"Dataset directory does not exist: {dataset}")
    archives = [archive_path(dataset, part) for part in parts]
    missing = [str(path) for path in archives if not path.is_file()]
    if missing:
        raise FileNotFoundError("Missing selected archives:\n" + "\n".join(missing))
    return archives


def _init_worker(args: argparse.Namespace, gpu_ids: list[int], next_worker: object) -> None:
    global _WORKER
    with next_worker.get_lock():
        slot = next_worker.value
        next_worker.value += 1
    gpu_id = gpu_ids[slot % len(gpu_ids)]
    import torch
    from sam2.build_sam import build_sam2_video_predictor_hf
    from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor

    torch.cuda.set_device(gpu_id)
    device = f"cuda:{gpu_id}"
    processor = AutoProcessor.from_pretrained(args.grounding_model)
    detector = AutoModelForZeroShotObjectDetection.from_pretrained(args.grounding_model).to(device).eval()
    predictor = build_sam2_video_predictor_hf(args.sam2_model, device=device)
    _WORKER = (args, device, processor, detector, predictor)


def _detect_people(frame, args, device, processor, detector):
    import torch
    from PIL import Image
    from torchvision.ops import nms

    image = Image.fromarray(frame).convert("RGB")
    inputs = processor(images=image, text="person.", return_tensors="pt").to(device)
    with torch.inference_mode():
        prediction = detector(**inputs)
    result = processor.post_process_grounded_object_detection(
        prediction,
        inputs.input_ids,
        threshold=args.box_threshold,
        text_threshold=args.text_threshold,
        target_sizes=[(image.height, image.width)],
    )[0]
    boxes = result["boxes"].detach().cpu()
    scores = result["scores"].detach().cpu()
    labels = result.get("text_labels", result.get("labels", []))
    valid = []
    for index, box in enumerate(boxes):
        if len(labels) > index and isinstance(labels[index], str) and "person" not in labels[index].lower():
            continue
        box[0::2].clamp_(0, image.width)
        box[1::2].clamp_(0, image.height)
        if box[2] > box[0] and box[3] > box[1]:
            valid.append(index)
    if not valid:
        return []
    keep = nms(boxes[valid], scores[valid], args.nms_threshold).tolist()
    return [
        {"box_xyxy": boxes[valid[i]].tolist(), "score": float(scores[valid[i]])}
        for i in keep
    ]


def _process_clip(clip: Clip) -> str:
    import decord
    import numpy as np
    import torch
    from PIL import Image

    args, device, processor, detector, predictor = _WORKER
    destination = output_path(args.output, clip.part, clip.clip_id)
    if (destination / "metadata.json").is_file():
        return "skipped"
    if destination.exists():
        raise FileExistsError(f"Incomplete output exists: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(prefix="person_video_") as video_tmp:
        video_path = Path(video_tmp) / "clip.mp4"
        video_path.write_bytes(clip.video)
        decord.bridge.set_bridge("native")
        reader = decord.VideoReader(str(video_path))
        if len(reader) == 0:
            raise ValueError(f"Video has no frames: {clip.source}")
        frame = reader[0].asnumpy()
        height, width = frame.shape[:2]
        frame_count = len(reader)
        del reader

        with torch.inference_mode():
            people = _detect_people(frame, args, device, processor, detector)
            metadata = {
                "source": clip.source,
                "prompt": "person",
                "frame_count": frame_count,
                "height": height,
                "width": width,
                "device": device,
                "grounding_model": args.grounding_model,
                "sam2_model": args.sam2_model,
                "people": [dict(person_id=i + 1, **item) for i, item in enumerate(people)],
                "status": "tracked" if people else "no_person",
            }
            with tempfile.TemporaryDirectory(prefix=f".{clip.clip_id}.", dir=destination.parent) as stage_name:
                stage = Path(stage_name)
                if people:
                    autocast = torch.autocast("cuda", dtype=torch.bfloat16) if torch.cuda.is_bf16_supported() else nullcontext()
                    with autocast:
                        state = predictor.init_state(
                            video_path=str(video_path),
                            offload_video_to_cpu=True,
                            offload_state_to_cpu=True,
                        )
                        try:
                            for person in metadata["people"]:
                                predictor.add_new_points_or_box(
                                    state,
                                    frame_idx=0,
                                    obj_id=person["person_id"],
                                    box=np.asarray(person["box_xyxy"], dtype=np.float32),
                                )
                                (stage / f"person_{person['person_id']:04d}").mkdir()
                            seen_frames = set()
                            expected_ids = {person["person_id"] for person in metadata["people"]}
                            for frame_idx, object_ids, mask_logits in predictor.propagate_in_video(state):
                                if frame_idx in seen_frames:
                                    raise RuntimeError(f"SAM 2 returned frame {frame_idx} more than once")
                                seen_frames.add(frame_idx)
                                if {int(person_id) for person_id in object_ids} != expected_ids:
                                    raise RuntimeError(f"SAM 2 returned incomplete person masks on frame {frame_idx}")
                                for index, person_id in enumerate(object_ids):
                                    mask = (mask_logits[index] > 0).squeeze().cpu().numpy().astype(np.uint8) * 255
                                    if mask.shape != (height, width):
                                        raise RuntimeError(f"Unexpected mask shape on frame {frame_idx}: {mask.shape}")
                                    Image.fromarray(mask, mode="L").save(
                                        stage / f"person_{int(person_id):04d}" / f"{frame_idx:06d}.png"
                                    )
                            if seen_frames != set(range(frame_count)):
                                raise RuntimeError(f"SAM 2 returned {len(seen_frames)} of {frame_count} frames")
                        finally:
                            predictor.reset_state(state)
                            del state
                (stage / "metadata.json").write_text(
                    json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
                )
                stage.rename(destination)
    return "tracked" if people else "no_person"


def run(args: argparse.Namespace) -> int:
    parts = range(args.start_part, args.end_part + 1)
    archives = validate_archives(args.dataset, parts)
    for archive in archives:
        print(archive, flush=True)
    if args.dry_run:
        return 0

    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; this script requires a GPU")
    visible_count = torch.cuda.device_count()
    gpu_ids = [int(item) for item in args.gpus.split(",")] if args.gpus else list(range(visible_count))
    if not gpu_ids or len(set(gpu_ids)) != len(gpu_ids) or any(i < 0 or i >= visible_count for i in gpu_ids):
        raise ValueError(f"--gpus must contain unique indices in 0..{visible_count - 1}")
    if args.workers > len(gpu_ids):
        print("Warning: multiple workers will share a GPU; check GPU memory", flush=True)
    args.output = args.output.resolve()
    args.output.mkdir(parents=True, exist_ok=True)

    context = mp.get_context("spawn")
    next_worker = context.Value("i", 0)
    counts = {"tracked": 0, "no_person": 0, "skipped": 0, "failed": 0}
    pending = {}
    with ProcessPoolExecutor(
        max_workers=args.workers,
        mp_context=context,
        initializer=_init_worker,
        initargs=(args, gpu_ids, next_worker),
    ) as pool:
        def collect_one() -> None:
            done, _ = wait(pending, return_when=FIRST_COMPLETED)
            for future in done:
                source = pending.pop(future)
                try:
                    result = future.result()
                except Exception as exc:
                    counts["failed"] += 1
                    print(f"FAILED {source}: {exc}", flush=True)
                else:
                    counts[result] += 1
                total = sum(counts.values())
                if total % 100 == 0:
                    print(f"Completed {total} clips: {counts}", flush=True)

        for part, archive in zip(parts, archives):
            for clip in iter_archive_clips(archive, part, args.output, counts):
                pending[pool.submit(_process_clip, clip)] = clip.source
                while len(pending) >= args.workers * 2:
                    collect_one()
        while pending:
            collect_one()
    print(f"Finished: {counts}", flush=True)
    return 1 if counts["failed"] else 0


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        return run(args)
    except (FileNotFoundError, ValueError, RuntimeError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
