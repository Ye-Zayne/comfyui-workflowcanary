# WorkflowCanary · 工作流更新后验收

Save a known-good image or video-frame batch as a local baseline, then compare future outputs after changing a model, custom node, parameter or environment. Fixed seeds and calibrated tolerances make the result useful.

保存一份认可的输出基线，在更新模型、节点或参数后比较尺寸、帧数、像素差异、指定区域和耗时。适合自己的常用工作流验收。

## Install / 安装

Clone into `ComfyUI/custom_nodes/comfyui-workflowcanary`, or install **WorkflowCanary** through ComfyUI Manager after publication. Restart ComfyUI. Only ComfyUI's existing torch, NumPy and Pillow dependencies are used. No model, cloud service, update rollback or package mutation.

## Nodes / 节点

| Node | What it does |
| --- | --- |
| `WorkflowCanaryCapture` | Saves lossless float32 NPZ output and an atomic JSON manifest, checksum and redacted provenance. Passes the images through. |
| `WorkflowCanaryCheck` | Compares outputs against that named baseline and returns baseline/current/heatmap IMAGE, difference MASK, report JSON and verdict. |

`baseline_root` defaults to `ComfyUI/output/workflowcanary`. Name baselines with letters, numbers, dots, underscores and hyphens; directory traversal is rejected. Capture refuses to overwrite by default. Enable `overwrite` only when you deliberately accept a new baseline. Each snapshot is immutable; overwritten manifests leave previous NPZ snapshots for recovery. A crash can leave `.capture.lock`; inspect the directory and remove that lock only after verifying no capture is active.

Check validates the snapshot checksum, shape, float32 IMAGE range and manifest structure. Manifest/snapshot symbolic links are rejected. Both output nodes always execute to notice changed baseline files rather than returning a cached comparison.

基线默认保存在输出目录。先用 Capture 存一次，再把工作流末尾切到 Check，使用同一名称。不要让 Capture 在每次生成时覆盖基线。

## Acceptance / 验收

- `2 = PASS`: compared output is inside thresholds.
- `1 = REVIEW`: baseline is missing, environment changed or supplied timing regressed.
- `0 = FAIL`: output shape differs, changed-pixel fraction is too large or SSIM is too low.

White in optional `roi` selects regions to compare; an empty overall ROI is rejected even when no baseline exists. Frames with empty individual ROIs are explicitly listed as skipped. Example: connect an inverted inpaint mask to compare the untouched area. SSIM uses a local 7×7 window and can include pixels neighboring the ROI. Dimensions and frame count are checked even with an ROI. The difference MASK and metrics obey the ROI; the heatmap shows differences across the entire image.

Images must be floating-point RGB/RGBA in `[0, 1]`; normalize HDR inputs explicitly. Baselines store float32, so higher-precision source tensors are compared at float32 precision.

`metadata_json` accepts your own structured values, for example:

```json
{"seed":42,"model_sha256":"actual-file-hash","node_versions":{"my-pack":"0.1.0"},"elapsed_seconds":12.5}
```

The report lists changed metadata, graph inputs and Python/torch/ComfyUI environment versions, including the source IMAGE dtype and tensor device. The image device is not an inferred sampler GPU. Model hashes and per-pack version declarations are supplied explicitly; they are not guessed. `elapsed_seconds` is measured by the caller or workflow runner, never inferred from sampler parameters; if provided, it must be nonnegative and finite (or null). JSON NaN/Infinity values are rejected before capture writes files. Keys matching token, secret, password, authorization, API key or credential are redacted before writing. Do not put other private data into custom metadata.

Change lists show correlations and cannot prove which update caused a regression. GPU, precision, backend and nondeterministic kernels can change output; maintain corresponding baselines and calibrate against normal repeat-run noise. This checks image/frame outputs; audio fidelity and automatic multi-workflow orchestration are outside v0.1.0.

## Example / 示例

Copy `examples/reference.png` and `changed.png` into the ComfyUI input folder. Submit `examples/capture_api.json` once, then submit `examples/check_api.json` using `POST /prompt` with the loaded JSON under the `prompt` key. These files are API prompts, not canvas workflow JSON. The capture example deliberately refuses overwrite; the check uses the modified image and reports a visual regression.

## Development

```sh
python -m pip install pytest torch numpy Pillow
python -m pytest tests --rootdir=.. --import-mode=importlib -q
```

MIT license. Version 0.1.0.
