import importlib.util
import json
import math
from pathlib import Path
import shutil

import pytest
import torch

spec = importlib.util.spec_from_file_location(
    "canary_nodes", Path(__file__).parents[1] / "nodes.py"
)
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)


def capture(root, image, **kwargs):
    return mod.WorkflowCanaryCapture().capture(image, "demo", str(root), **kwargs)


def check(root, image, **kwargs):
    return mod.WorkflowCanaryCheck().check(image, "demo", str(root), **kwargs)["result"]


def test_identical_pass_changed_fail_and_no_baseline_review(tmp_path):
    image = torch.rand(1, 8, 8, 3)
    assert check(tmp_path, image)[-1] == 1
    capture(tmp_path, image)
    assert check(tmp_path, image)[-1] == 2
    assert check(tmp_path, torch.zeros_like(image))[-1] == 0


def test_shape_mismatch_and_corruption(tmp_path):
    image = torch.zeros(1, 8, 8, 3)
    capture(tmp_path, image)
    assert check(tmp_path, torch.zeros(2, 8, 8, 3))[-1] == 0
    snapshot = next((tmp_path / "demo").glob("*.npz"))
    snapshot.write_bytes(b"corrupt")
    with pytest.raises(ValueError, match="hash mismatch"):
        check(tmp_path, image)


def test_protected_roi_and_json_metadata_changes(tmp_path):
    image = torch.zeros(1, 16, 16, 3)
    capture(tmp_path, image, metadata_json='{"model_hash":"before"}')
    edited = image.clone()
    edited[:, 10:] = 1
    roi = torch.zeros(1, 16, 16)
    roi[:, :3] = 1
    result = check(tmp_path, edited, roi=roi, metadata_json='{"model_hash":"after"}')
    assert result[-1] == 2
    assert any(
        x["path"] == "metadata.model_hash" for x in json.loads(result[2])["changes"]
    )


def test_time_slowdown_and_secret_redaction(tmp_path):
    image = torch.zeros(1, 8, 8, 3)
    capture(
        tmp_path,
        image,
        metadata_json='{"elapsed_seconds":10,"api_key":"never-persist"}',
    )
    assert "never-persist" not in (tmp_path / "demo" / "baseline.json").read_text()
    assert check(tmp_path, image, metadata_json='{"elapsed_seconds":25}')[-1] == 1


def test_overwrite_and_path_traversal(tmp_path):
    image = torch.zeros(1, 8, 8, 3)
    capture(tmp_path, image)
    with pytest.raises(ValueError, match="already exists"):
        capture(tmp_path, image)
    capture(tmp_path, image, overwrite=True)
    with pytest.raises(ValueError, match="safe filename"):
        mod.WorkflowCanaryCapture().capture(image, "../escape", str(tmp_path))


def test_empty_roi_rejected(tmp_path):
    image = torch.zeros(1, 8, 8, 3)
    capture(tmp_path, image)
    with pytest.raises(ValueError, match="Empty ROI"):
        check(tmp_path, image, roi=torch.zeros(1, 8, 8))


@pytest.mark.parametrize("metadata", ['{"x":NaN}', '{"x":[Infinity]}', '{"x":1e400}'])
def test_nonfinite_json_rejected_before_capture_creates_snapshot(tmp_path, metadata):
    image = torch.zeros(1, 8, 8, 3)
    with pytest.raises(ValueError, match="non-finite"):
        capture(tmp_path, image, metadata_json=metadata)
    assert not (tmp_path / "demo").exists()
    with pytest.raises(ValueError, match="non-finite"):
        check(tmp_path, image, metadata_json=metadata)


@pytest.mark.parametrize("elapsed", [-1, True, "unmeasured"])
def test_invalid_timing_metadata_rejected(tmp_path, elapsed):
    with pytest.raises(ValueError, match="elapsed_seconds"):
        capture(
            tmp_path,
            torch.zeros(1, 8, 8, 3),
            metadata_json=json.dumps({"elapsed_seconds": elapsed}),
        )


def test_empty_roi_rejected_without_baseline_and_empty_frames_are_reported(tmp_path):
    image = torch.zeros(2, 8, 8, 3)
    with pytest.raises(ValueError, match="Empty ROI"):
        check(tmp_path, image, roi=torch.zeros(1, 8, 8))
    capture(tmp_path, image)
    changed = image.clone()
    changed[1] = 1
    roi = torch.zeros(2, 8, 8)
    roi[0] = 1
    result = check(tmp_path, changed, roi=roi)
    report = json.loads(result[2])
    assert result[-1] == 2
    assert report["frames"][1]["skipped"] == "empty ROI for this frame"
    assert not bool(result[1].any())


def test_symbolic_link_snapshot_and_manifest_are_rejected(tmp_path):
    image = torch.zeros(1, 8, 8, 3)
    capture(tmp_path, image)
    folder = tmp_path / "demo"
    snapshot = next(folder.glob("*.npz"))
    outside = tmp_path / "outside.npz"
    shutil.copyfile(snapshot, outside)
    snapshot.unlink()
    snapshot.symlink_to(outside)
    with pytest.raises(ValueError, match="symbolic link"):
        check(tmp_path, image)
    snapshot.unlink()
    shutil.copyfile(outside, snapshot)
    manifest = folder / "baseline.json"
    external_manifest = tmp_path / "external.json"
    shutil.copyfile(manifest, external_manifest)
    manifest.unlink()
    manifest.symlink_to(external_manifest)
    with pytest.raises(ValueError, match="symbolic link"):
        check(tmp_path, image)


def test_manifest_structure_and_nonfinite_numbers_are_rejected(tmp_path):
    image = torch.zeros(1, 8, 8, 3)
    capture(tmp_path, image)
    manifest = tmp_path / "demo" / "baseline.json"
    manifest.write_text("[]")
    with pytest.raises(ValueError, match="manifest schema"):
        check(tmp_path, image)
    manifest.write_text('{"schema":"workflowcanary/1","bad":NaN}')
    with pytest.raises(ValueError, match="non-finite"):
        check(tmp_path, image)


def test_source_dtype_is_provenance_and_changes_need_review(tmp_path):
    image = torch.zeros(1, 8, 8, 3)
    capture(tmp_path, image)
    result = check(tmp_path, image.half())
    report = json.loads(result[2])
    assert result[-1] == 1
    assert report["environment"]["image_dtype"] == "torch.float16"
    assert any(item["path"] == "environment.image_dtype" for item in report["changes"])


def test_output_nodes_do_not_cache_mutable_baseline_files():
    assert math.isnan(mod.WorkflowCanaryCapture.IS_CHANGED())
    assert math.isnan(mod.WorkflowCanaryCheck.IS_CHANGED())


def test_out_of_range_images_are_rejected_before_float32_overflow(tmp_path):
    image = torch.full((1, 8, 8, 3), 1e300, dtype=torch.float64)
    with pytest.raises(ValueError, match=r"within \[0, 1\]"):
        capture(tmp_path, image)
