"""Local, explicit output baselines. No model loading or environment mutation."""

import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import platform
import re
import tempfile
import time
import uuid

import numpy as np
import torch
import torch.nn.functional as F


SECRET_NAMES = re.compile(
    r"token|secret|password|passwd|authorization|api.?key|credential", re.I
)


def scrub(value):
    if isinstance(value, dict):
        return {
            str(k): "<redacted>" if SECRET_NAMES.search(str(k)) else scrub(v)
            for k, v in value.items()
        }
    if isinstance(value, list):
        return [scrub(x) for x in value]
    return value


def finite_json(value, label):
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError(f"{label} contains a non-finite JSON number.")
    if isinstance(value, dict):
        for key, child in value.items():
            finite_json(child, f"{label}.{key}")
    elif isinstance(value, (list, tuple)):
        for index, child in enumerate(value):
            finite_json(child, f"{label}[{index}]")


def decode_json(text, label):
    try:
        value = json.loads(text)
    except (json.JSONDecodeError, TypeError) as exc:
        raise ValueError(f"{label} must contain valid JSON.") from exc
    finite_json(value, label)
    return value


def image_array(images):
    if not isinstance(images, torch.Tensor):
        raise ValueError("Expected a ComfyUI IMAGE tensor.")
    if images.ndim != 4 or images.shape[-1] not in (3, 4) or min(images.shape) < 1:
        raise ValueError("Expected nonempty IMAGE [batch, height, width, 3 or 4].")
    if not bool(torch.isfinite(images).all()):
        raise ValueError("Cannot compare or capture NaN/infinite images.")
    if not images.is_floating_point() or bool(((images < 0) | (images > 1)).any()):
        raise ValueError(
            "Images must be floating-point RGB/RGBA within [0, 1]; normalize HDR inputs explicitly."
        )
    return images.detach().cpu().float().numpy()


def baseline_dir(name, root):
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,95}", name) or name in (
        ".",
        "..",
    ):
        raise ValueError(
            "baseline_name must be a safe filename (letters, numbers, ., _, -)."
        )
    if not root.strip():
        import folder_paths

        root = str(Path(folder_paths.get_output_directory()) / "workflowcanary")
    base = Path(root).expanduser().resolve()
    target = (base / name).resolve()
    if target.parent != base:
        raise ValueError("Baseline directory cannot escape baseline_root.")
    return target


def atomic_json(path, data):
    fd, temp = tempfile.mkstemp(dir=path.parent, prefix=".canary-", suffix=".json")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2, allow_nan=False)
            f.flush()
            os.fsync(f.fileno())
        os.replace(temp, path)
    finally:
        if os.path.exists(temp):
            os.unlink(temp)


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def environment():
    result = {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "platform": platform.system(),
        "machine": platform.machine(),
        "cuda": torch.version.cuda,
    }
    try:
        import comfyui_version

        result["comfyui"] = comfyui_version.__version__
    except ImportError:
        result["comfyui"] = None
    try:
        result["numpy"] = importlib.metadata.version("numpy")
    except importlib.metadata.PackageNotFoundError:
        result["numpy"] = None
    return result


def provenance(metadata_json, prompt, images=None):
    metadata = decode_json(metadata_json or "{}", "metadata_json")
    if not isinstance(metadata, dict):
        raise ValueError(
            "metadata_json must be a JSON object (seed, model hashes, node versions, elapsed_seconds...)."
        )
    elapsed = metadata.get("elapsed_seconds")
    if elapsed is not None and (
        isinstance(elapsed, bool)
        or not isinstance(elapsed, (int, float))
        or elapsed < 0
    ):
        raise ValueError(
            "metadata_json.elapsed_seconds must be a nonnegative finite number or null."
        )
    graph = {
        str(k): v
        for k, v in (prompt or {}).items()
        if isinstance(v, dict)
        and not str(v.get("class_type", "")).startswith("WorkflowCanary")
    }
    info = {
        "environment": environment(),
        "metadata": scrub(metadata),
        "workflow": scrub(graph),
    }
    if images is not None:
        info["environment"].update(
            {"image_dtype": str(images.dtype), "image_device": str(images.device)}
        )
    finite_json(info, "provenance")
    return info


def load_baseline(folder):
    path = folder / "baseline.json"
    if path.is_symlink():
        raise ValueError("Baseline manifest cannot be a symbolic link.")
    manifest = decode_json(path.read_text(encoding="utf-8"), "baseline manifest")
    if not isinstance(manifest, dict) or manifest.get("schema") != "workflowcanary/1":
        raise ValueError("Unsupported baseline manifest schema.")
    snapshot_name = manifest.get("snapshot", "")
    if not isinstance(snapshot_name, str) or not re.fullmatch(
        r"[0-9a-f]{32}\.npz", snapshot_name
    ):
        raise ValueError("Unsafe baseline snapshot filename.")
    checksum = manifest.get("sha256")
    if not isinstance(checksum, str) or not re.fullmatch(r"[0-9a-f]{64}", checksum):
        raise ValueError("Invalid baseline snapshot checksum.")
    shape = manifest.get("shape")
    if (
        not isinstance(shape, list)
        or len(shape) != 4
        or any(type(n) is not int or n < 1 for n in shape)
        or shape[-1] not in (3, 4)
    ):
        raise ValueError("Invalid baseline image shape.")
    info = manifest.get("provenance")
    if not isinstance(info, dict) or any(
        not isinstance(info.get(key), dict)
        for key in ("environment", "metadata", "workflow")
    ):
        raise ValueError("Invalid baseline provenance.")
    snapshot = folder / snapshot_name
    if snapshot.is_symlink() or snapshot.resolve().parent != folder.resolve():
        raise ValueError(
            "Baseline snapshot cannot be a symbolic link or escape its directory."
        )
    if sha256(snapshot) != checksum:
        raise ValueError("Baseline snapshot hash mismatch; baseline is damaged.")
    try:
        with np.load(snapshot, allow_pickle=False) as stored:
            baseline = stored["images"]
    except (ValueError, OSError, KeyError) as exc:
        raise ValueError("Baseline snapshot is not a readable image archive.") from exc
    if (
        list(baseline.shape) != shape
        or baseline.dtype != np.float32
        or not np.isfinite(baseline).all()
        or np.any((baseline < 0) | (baseline > 1))
    ):
        raise ValueError(
            "Baseline content does not match its manifest or valid float32 IMAGE range."
        )
    return manifest, baseline


def differences(a, b, path="", limit=80):
    out = []
    if isinstance(a, dict) and isinstance(b, dict):
        for key in sorted(set(a) | set(b)):
            label = f"{path}.{key}" if path else key
            if key not in a or key not in b:
                out.append({"path": label, "before": a.get(key), "after": b.get(key)})
            else:
                out.extend(differences(a[key], b[key], label, limit))
            if len(out) >= limit:
                break
    elif a != b:
        out.append({"path": path, "before": a, "after": b})
    return out[:limit]


def roi_mask(mask, shape):
    if mask is None:
        return np.ones(shape[:3], dtype=bool)
    array = mask.detach().cpu().float().numpy()
    if array.ndim == 2:
        array = array[None]
    if (
        array.ndim != 3
        or array.shape[1:] != shape[1:3]
        or array.shape[0] not in (1, shape[0])
    ):
        raise ValueError(
            "ROI must match image dimensions; batch may be 1 or image batch size."
        )
    if not np.isfinite(array).all() or np.any((array < 0) | (array > 1)):
        raise ValueError("ROI must be finite and within [0, 1].")
    array = np.broadcast_to(array, shape[:3]) > 0.5
    if not array.any():
        raise ValueError("Empty ROI cannot produce a meaningful verdict.")
    return array


def ssim_map(a, b):
    a = torch.from_numpy(a.copy()).permute(0, 3, 1, 2)
    b = torch.from_numpy(b.copy()).permute(0, 3, 1, 2)
    k = min(7, a.shape[-2], a.shape[-1])
    k -= k % 2 == 0
    pool = lambda x: F.avg_pool2d(x, k, 1, k // 2, count_include_pad=False)
    ma, mb = pool(a), pool(b)
    va = (pool(a * a) - ma * ma).clamp_min(0)
    vb = (pool(b * b) - mb * mb).clamp_min(0)
    cov = pool(a * b) - ma * mb
    score = ((2 * ma * mb + 0.01**2) * (2 * cov + 0.03**2)) / (
        (ma * ma + mb * mb + 0.01**2) * (va + vb + 0.03**2)
    )
    return score.mean(1).clamp(-1, 1).numpy()


class WorkflowCanaryCapture:
    CATEGORY = "WorkflowCanary"
    FUNCTION = "capture"
    OUTPUT_NODE = True
    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("images", "baseline_manifest")

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
                "baseline_name": ("STRING", {"default": "my-workflow"}),
                "baseline_root": ("STRING", {"default": ""}),
                "metadata_json": ("STRING", {"default": "{}", "multiline": True}),
                "overwrite": ("BOOLEAN", {"default": False}),
            },
            "hidden": {"prompt": "PROMPT"},
        }

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        return float("nan")

    def capture(
        self,
        images,
        baseline_name="my-workflow",
        baseline_root="",
        metadata_json="{}",
        overwrite=False,
        prompt=None,
    ):
        array = image_array(images)
        info = provenance(metadata_json, prompt, images)
        folder = baseline_dir(baseline_name, baseline_root)
        folder.mkdir(parents=True, exist_ok=True)
        manifest_path = folder / "baseline.json"
        # O_EXCL prevents two simultaneous captures from publishing conflicting snapshots.
        lock = folder / ".capture.lock"
        try:
            fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
        except FileExistsError as exc:
            raise ValueError(
                "Another capture is active, or a crashed capture left .capture.lock; inspect before removing it."
            ) from exc
        os.close(fd)
        try:
            if manifest_path.exists() and not overwrite:
                raise ValueError(
                    "Baseline already exists. Use another name or explicitly enable overwrite."
                )
            snapshot = folder / (uuid.uuid4().hex + ".npz")
            temp = snapshot.with_suffix(".tmp")
            try:
                with temp.open("wb") as stream:
                    np.savez_compressed(stream, images=array)
                    stream.flush()
                    os.fsync(stream.fileno())
                os.replace(temp, snapshot)
            finally:
                temp.unlink(missing_ok=True)
            manifest = {
                "schema": "workflowcanary/1",
                "created_at": time.time(),
                "name": baseline_name,
                "shape": list(array.shape),
                "snapshot": snapshot.name,
                "sha256": sha256(snapshot),
                "provenance": info,
            }
            atomic_json(manifest_path, manifest)
        finally:
            lock.unlink(missing_ok=True)
        return {
            "ui": {"text": [str(manifest_path)]},
            "result": (images, str(manifest_path)),
        }


class WorkflowCanaryCheck:
    CATEGORY = "WorkflowCanary"
    FUNCTION = "check"
    OUTPUT_NODE = True
    RETURN_TYPES = ("IMAGE", "MASK", "STRING", "INT")
    RETURN_NAMES = ("comparison", "difference_mask", "report_json", "verdict")

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
                "baseline_name": ("STRING", {"default": "my-workflow"}),
                "baseline_root": ("STRING", {"default": ""}),
                "metadata_json": ("STRING", {"default": "{}", "multiline": True}),
                "pixel_tolerance": ("FLOAT", {"default": 0.03, "min": 0.0, "max": 1.0}),
                "max_changed_fraction": (
                    "FLOAT",
                    {"default": 0.05, "min": 0.0, "max": 1.0},
                ),
                "min_ssim": ("FLOAT", {"default": 0.95, "min": -1.0, "max": 1.0}),
                "max_time_ratio": ("FLOAT", {"default": 1.5, "min": 1.0, "max": 100.0}),
            },
            "optional": {"roi": ("MASK",)},
            "hidden": {"prompt": "PROMPT"},
        }

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        return float("nan")

    def check(
        self,
        images,
        baseline_name="my-workflow",
        baseline_root="",
        metadata_json="{}",
        pixel_tolerance=0.03,
        max_changed_fraction=0.05,
        min_ssim=0.95,
        max_time_ratio=1.5,
        roi=None,
        prompt=None,
    ):
        for value, low, high in [
            (pixel_tolerance, 0, 1),
            (max_changed_fraction, 0, 1),
            (min_ssim, -1, 1),
            (max_time_ratio, 1, 100),
        ]:
            if not math.isfinite(value) or not low <= value <= high:
                raise ValueError("Invalid comparison threshold.")
        array = image_array(images)
        current = provenance(metadata_json, prompt, images)
        selected = roi_mask(roi, array.shape)
        folder = baseline_dir(baseline_name, baseline_root)
        path = folder / "baseline.json"
        report = {
            "schema": "workflowcanary/1",
            "baseline": baseline_name,
            "issues": [],
            "changes": [],
            "frames": [],
            "current_shape": list(array.shape),
            "environment": current["environment"],
        }
        verdict = 2
        preview = images[..., :3].detach().cpu().float().clone()
        difference = torch.zeros(images.shape[:3], dtype=torch.float32)
        if path.is_symlink():
            raise ValueError("Baseline manifest cannot be a symbolic link.")
        if not path.exists():
            verdict = 1
            report["issues"].append("NO_BASELINE: capture a known-good output first.")
        else:
            manifest, baseline = load_baseline(folder)
            report["baseline_shape"] = list(baseline.shape)
            report["changes"] = differences(manifest["provenance"], current)
            if baseline.shape != array.shape:
                verdict = 0
                report["issues"].append(
                    {
                        "type": "SHAPE_MISMATCH",
                        "baseline": list(baseline.shape),
                        "current": list(array.shape),
                    }
                )
            else:
                error = np.abs(array - baseline)
                changed = error.max(-1) > pixel_tolerance
                difference = torch.from_numpy((changed & selected).astype(np.float32))
                scores = ssim_map(baseline, array)
                for i in range(array.shape[0]):
                    region = selected[i]
                    if not region.any():
                        report["frames"].append(
                            {"index": i, "skipped": "empty ROI for this frame"}
                        )
                        continue
                    fraction = float(changed[i][region].mean())
                    score = float(scores[i][region].mean())
                    mae = float(error[i][region].mean())
                    mse = float(np.square(error[i][region], dtype=np.float64).mean())
                    fail = fraction > max_changed_fraction or score < min_ssim
                    if fail:
                        verdict = 0
                    report["frames"].append(
                        {
                            "index": i,
                            "changed_fraction": fraction,
                            "ssim": score,
                            "mae": mae,
                            "psnr_db": None if mse == 0 else -10 * math.log10(mse),
                            "failed": fail,
                        }
                    )
                heat = np.stack(
                    [
                        error.max(-1).clip(0, 1),
                        np.zeros(array.shape[:3]),
                        np.zeros(array.shape[:3]),
                    ],
                    -1,
                )
                preview = torch.from_numpy(
                    np.concatenate(
                        [baseline[..., :3], array[..., :3], heat], axis=2
                    ).astype(np.float32)
                )
            old_env = manifest["provenance"]["environment"]
            if old_env != current["environment"]:
                verdict = min(verdict, 1)
                report["issues"].append(
                    "ENVIRONMENT_CHANGED: review or establish a baseline for this hardware/software stack."
                )
            before = manifest["provenance"]["metadata"].get("elapsed_seconds")
            after = current["metadata"].get("elapsed_seconds")
            if (
                isinstance(before, (int, float))
                and isinstance(after, (int, float))
                and before > 0
                and after >= 0
            ):
                if not math.isfinite(before) or not math.isfinite(after):
                    raise ValueError("elapsed_seconds must be finite.")
                report["time_ratio"] = after / before
                if after / before > max_time_ratio:
                    verdict = min(verdict, 1)
                    report["issues"].append(
                        "SLOWDOWN: elapsed_seconds exceeds the configured time ratio."
                    )
        report.update(
            {
                "verdict_code": verdict,
                "verdict": ["FAIL", "REVIEW", "PASS"][verdict],
                "notes": [
                    "Changes are correlations, not proven causes.",
                    "Use fixed seeds and calibrate thresholds against normal repeat-run variation.",
                    "Timing is supplied explicitly as metadata elapsed_seconds; no sampler timing is inferred.",
                    "Model hashes and node versions may be supplied through metadata_json; no expensive model hashing runs implicitly.",
                ],
            }
        )
        encoded = json.dumps(report, ensure_ascii=False, allow_nan=False)
        return {
            "ui": {"text": [encoded]},
            "result": (preview, difference, encoded, verdict),
        }
