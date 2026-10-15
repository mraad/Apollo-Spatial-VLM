# Model evaluation (RTX PRO 6000 server, 2026-10-14)

Is there a better depth model than `DA3METRIC-LARGE`, and a better VLM than Qwen3-VL for picking targets? Both were measured against public ground truth on the RTX PRO 6000 server (4 × RTX PRO 6000 Blackwell, 96 GB each): one VLM per GPU on GPUs 0–2, depth models on GPU 3. Raw per-sample results are in `results/`.

## VLM: pointing at a referred object

**Data:** 200 seeded RefCOCO val referring expressions (`jxu124/refcoco` annotations, COCO train2014 images).

**Score:** a hit means the model's point falls inside the ground-truth box. *Parsed accuracy* counts only answers the parser could read, so it separates pointing skill from format failures.

**Setup:**
- Images go at the demo's 384 px, and also at 1024 px.
- Every model ran on vLLM 0.31 in eager mode, BF16, with one request at a time; latency is the median per request.
- Qwen3.6 ran with thinking off.
- Two prompt styles were tested:
  - **demo:** the demo's own JSON point format (`custom_prompt_template` and `parse_vlm_targets`), scoring the highest-confidence target.
  - **native:** the model's own grounding output. For Qwen that's a JSON box, scored at its centre; Molmo2 is always asked "Point to …".

| Model | Prompt | Acc. 384 px | Acc. 1024 px | Unparsed answers (384 / 1024) | Parsed acc. (384 / 1024) | Latency at 384 px |
|---|---|---|---|---|---|---|
| Qwen3-VL-32B-Instruct (current) | demo | 0.760 | 0.630 | 31 / 62 | 0.899 / 0.913 | 1.39 s |
| Qwen3-VL-32B-Instruct | native | 0.865 | **0.905** | 4 / 1 | 0.883 / 0.910 | 1.62 s |
| Qwen3.6-27B | demo | 0.860 | **0.905** | 8 / 2 | 0.896 / 0.914 | 1.55 s |
| Qwen3.6-27B | native | **0.905** | 0.875 | 1 / 0 | 0.910 / 0.875 | 2.35 s |
| Molmo2-8B | native | **0.895** | 0.875 | 4 / 4 | 0.913 / 0.893 | **0.52 s** |

Findings:
- **Pointing skill is the same across models.** Once an answer parses, every model lands at 0.88–0.91, within the ±4% margin (95%) that 200 samples allow.
- **The demo's prompt format is the bottleneck.** Qwen3-VL-32B wants to answer with a box. Forced into the demo's single x/y format, it writes invalid JSON such as `"x":150,820,"y":900` in 15% of answers at 384 px and 31% at 1024 px. Asking for its native box raises it from 0.76 to 0.865 (384 px) and from 0.63 to 0.905 (1024 px).
- **Qwen3.6-27B follows the demo's format far better** (8 unparsed answers vs 31 at 384 px), so it would be a drop-in upgrade. The demo would need to send `chat_template_kwargs: {"enable_thinking": false}`, because thinking is on by default.
- **Molmo2-8B matches the best accuracy at 2.7× the speed** with a quarter of the parameters. It needs its own prompt and parser in the demo (`<points coords="1 1 x y">`, also on the 0–1000 grid). Its card notes training data meant for research use only.
- **Larger images don't help consistently**, so the demo's 384 px is fine.
- **Demo bug found and fixed in `demo.py`:** malformed VLM JSON made `parse_vlm_targets` raise, turning the request into a 500 error. It now counts as "no targets", with a unit test.

## Depth: metric depth against laser ground truth

**Data:** all 771 DIODE val frames (325 indoor, 446 outdoor, 1024×768).

**Metrics:** AbsRel (lower is better) and δ1 (the share of pixels within ×1.25 of the truth, higher is better), on raw metric output. "Scale" is the median of predicted ÷ true depth (1.00 is right). The *aligned* columns rescale each prediction by its median first, which measures shape regardless of overall scale.

**Field of view:** DIODE's camera has a 60.0° horizontal field of view, exactly the demo's assumption, so full frames flatter DA3. Centre crops (half width and height, a 32.2° view) test a wrong guess against known ground truth. Three variants were compared:
- `da3`: the demo's pipeline with the 60° guess.
- `da3 + true focal`: the real focal length, as EXIF would supply.
- MoGe-3: estimates the field of view itself.

**Full frames (60° view):**

| Model | AbsRel | δ1 | Scale | Aligned AbsRel | Aligned δ1 | Outdoor δ1 | Latency |
|---|---|---|---|---|---|---|---|
| DA3METRIC-LARGE (60° guess, current) | 0.488 | 0.117 | 0.72 | 0.303 | 0.786 | 0.130 | **22 ms** |
| DA3METRIC-LARGE + true focal | 0.480 | 0.136 | 0.73 | 0.303 | 0.786 | 0.145 | 22 ms |
| MoGe-3 ViT-L | **0.332** | **0.529** | **0.91** | **0.216** | 0.819 | **0.598** | 161 ms |
| MoGe-3 ViT-G | 0.347 | 0.510 | 0.91 | 0.217 | **0.824** | 0.578 | 356 ms |

**Centre crops (32.2° view):**

| Model | AbsRel | δ1 | Scale | Predicted FOV (indoor / outdoor) |
|---|---|---|---|---|
| DA3METRIC-LARGE (60° guess, current) | 0.651 | 0.027 | 0.39 | (assumes 60°) |
| DA3METRIC-LARGE + true focal | **0.428** | **0.305** | **0.80** | (given) |
| MoGe-3 ViT-L | 0.454 | 0.247 | 0.75 | 47.3° / 35.2° |
| MoGe-3 ViT-G | 0.465 | 0.261 | 0.76 | 47.6° / 35.8° |

Findings:
- **MoGe-3 ViT-L is the stronger metric depth model.** On full frames it gets δ1 0.53 vs 0.12 for DA3 and nearly correct scale (0.91, and 1.00 outdoors). It also has the better shape (aligned AbsRel 0.216 vs 0.303), mostly outdoors. DA3METRIC underestimates distance by about 28% even when the field of view is right.
- **Knowing the true focal length matters most.** With the 60° guess on a 32° view, DA3 collapses (δ1 0.03, depths at 0.39×). Given the true focal length it recovers to δ1 0.31, on par with MoGe-3. This supports the EXIF idea in `docs/exif-geolocation.md`.
- **MoGe-3's field-of-view estimate is good outdoors but weak indoors on crops** (47° predicted for a true 32°). Passing an EXIF field of view to MoGe (`fov_x`) is untested.
- **ViT-G adds nothing over ViT-L** and is 2.2× slower.
- **Harbour bridge (SOH sample):** DA3 puts it at 18.48 m with either focal length. MoGe-3 marks it invalid (no depth) rather than guessing.

## CUDA graphs (2026-10-14)

`run-vlm.sh` used to pass `--enforce-eager`. With CUDA graphs on the RTX PRO 6000 server, Qwen3.6-27B (demo prompt, 384 px) kept its accuracy and got faster:

| Mode | Accuracy | Unparsed answers | Median latency |
|---|---|---|---|
| Eager (`vlm-qwen3.6-27b-384.json`) | 0.860 | 8 | 1.55 s |
| CUDA graphs (`vlm-qwen3.6-27b-graphs-384.json`) | 0.860 | 6 | **1.13 s** |

The full demo request (three targets, 77 output tokens) went from 3.85 to 2.85 s. vLLM's first start compiles kernels (166 s vs 55 s eager), and later starts take ~65 s. On a Jetson AGX Orin the gain was only 7.35 → 7.17 s per request, while startup doubled (110 → 221 s), so Orin keeps eager mode.

## FP8 weights (2026-10-14)

`Qwen/Qwen3.6-27B-FP8` on the RTX PRO 6000 server (CUDA graphs; vLLM picked native `CutlassFp8BlockScaledMMKernel`), demo prompt, 384 px:

| Weights | Accuracy | Unparsed answers | Median latency | Demo request (3 targets) |
|---|---|---|---|---|
| BF16 (`vlm-qwen3.6-27b-graphs-384.json`) | 0.860 | 6 | 1.13 s | 2.85 s |
| FP8 (`vlm-qwen3.6-27b-fp8-384.json`) | 0.870 | 6 | **0.72 s** | **1.73 s** |

No accuracy cost at this sample size (±4%), 31 GB of weights instead of 56 GB. On a Jetson AGX Orin (no native FP8) vLLM falls back to Marlin weight-only FP8: ~16 s per demo request, so Molmo2-8B is the better VLM there.

## Model choices in the current demo

1. **VLM:** Qwen3.6-27B follows the demo format more reliably; the demo already sends `enable_thinking: false`. FP8 is faster on Blackwell. Molmo2-8B has a dedicated prompt/parser path and is the tested choice on Orin. Configure endpoints explicitly; the setup default remains Qwen3-VL-8B.
2. **Depth:** MoGe-3 ViT-L is available on Linux/CUDA with `WITH_MOGE=1 ./setup.sh`; select it in the page or set `DEPTH_MODEL=moge3-l`. DA3 remains the default and the macOS path. ViT-G is also available but did not improve these aggregate accuracy scores.
3. **Field of view:** the demo uses EXIF focal length when available for either depth model. The benchmark above did not test passing EXIF FOV into MoGe.

## Caveats

- One depth dataset (DIODE), 200 RefCOCO samples, one run each. vLLM isn't fully deterministic even at temperature 0.
- RefCOCO asks for one referred object; the demo usually asks for several targets per prompt.
- The baseline tables use eager mode, one request at a time, on Blackwell GPUs. The CUDA-graph and FP8 sections state their different settings; current launch scripts default to CUDA graphs unless `ENFORCE_EAGER=1`.
- The native Qwen runs were re-scored from their stored raw answers after the parser learned the `"bbox"` key Qwen3-VL-32B uses. This is identical to re-running with the fixed parser.

## Reproduce (RTX PRO 6000 server)

Cache the model weights before using `eval/serve.sh` (it runs offline). These commands reproduce the eager baseline; omit `ENFORCE_EAGER=1` for the CUDA-graph comparison.

```bash
uv run eval/prepare_refcoco.py                    # 200 RefCOCO refs + their COCO images -> eval/data/
curl -O http://diode-dataset.s3.amazonaws.com/val.tar.gz && tar xzf val.tar.gz -C eval/data   # DIODE val
ENFORCE_EAGER=1 eval/serve.sh 0 8012 qwen3-vl-32b Qwen/Qwen3-VL-32B-Instruct
ENFORCE_EAGER=1 eval/serve.sh 1 8014 qwen3.6-27b Qwen/Qwen3.6-27B
ENFORCE_EAGER=1 eval/serve.sh 2 8013 molmo2-8b allenai/Molmo2-8B --max-num-batched-tokens 16384
.venv/bin/python eval/vlm_eval.py --url http://127.0.0.1:8014/v1 --model qwen3.6-27b --name qwen3.6-27b --no-think  # optionally add --prompt native and/or --max-dim 1024
.venv/bin/python eval/vlm_eval.py --url http://127.0.0.1:8013/v1 --model molmo2-8b --name molmo2-8b --kind molmo
CUDA_VISIBLE_DEVICES=3 .venv/bin/python eval/depth_eval.py --model da3        # or da3-gt-focal
uv venv --python 3.12 eval/.venv-moge && uv pip install --python eval/.venv-moge --torch-backend auto torch torchvision git+https://github.com/microsoft/MoGe.git
CUDA_VISIBLE_DEVICES=3 eval/.venv-moge/bin/python eval/depth_eval.py --model moge3-l   # or moge3-g
```

## Release 0.2.1 validation (2026-10-15)

Both gc1 (RTX PRO 6000 Blackwell) and Jetson AGX Orin 64 GB passed **16 unit tests**, **six API inference cases** and **three invalid requests**. The API run covered DA3, MoGe-3 L and G, each with and without the single-target prompt `locate the Sydney Opera House`. gc1 used Qwen3-VL-32B; Orin used Molmo2-8B with the [tested launch settings](../README.md#jetson-agx-orin-64-gb).

The smoke test verifies the advertised version/models, decoded image dimensions, a finite nonnegative float16 depth grid with valid pixels, the presence of prompted 2D/3D targets, finite 3D coordinates and 400 responses for invalid model names/image data. Run it against an already running demo and VLM:

```bash
.venv/bin/python -m unittest discover -s tests -v
.venv/bin/python tests/smoke.py --url http://127.0.0.1:8080
```

Orin's latest individual request times (the first MoGe calls include lazy model loading; DA3 loads at server startup):

| Depth model | No prompt, first call | Single-target prompt, warm model |
|---|---|---|
| DA3 | 1.331 s | 3.343 s |
| MoGe-3 L | 14.390 s | 4.249 s |
| MoGe-3 G | 42.724 s | 4.486 s |

These are single smoke measurements, not benchmark medians, and use a different prompt/configuration from the historical three-target timings in the root README. This release did not rerun the RefCOCO or DIODE accuracy benchmarks. A separate WebGL check confirmed that labels draw after Gaussian splats and remain visible from front and side views.

The committed [gc1 0.2.0](results/smoke-gc1-0.2.0.json) and [Orin 0.2.0](results/smoke-orin-0.2.0.json) records remain historical. The gc1 record also measures the grayscale preview optimization: median preview encoding fell from 10.334 to 7.527 ms and the PNG from 25,182 to 17,111 bytes, with identical valid-pixel values. Warm depth-only latency changed from 57.0 to 53.4 ms; prompted latency was effectively unchanged (1.5581 to 1.5531 s).

## Two-view localization and Gaussian fusion research (2026-10-15)

The [proposal](../docs/two-view-localization.md) combines calibrated two-view measurement with a required final output of **one fused Gaussian-splat scene**. It covers camera/GPS observability, height references, model choices, the bundled DA3 two-pose alignment failure, surface fusion, and the separate role of generative diffusion.

The [CPU experiment](two_view_localization.py) ran 150,000 synthetic trials and a calibrated Middlebury `Motorcycle-perfect` stereo pair. Its [saved results](results/two-view-localization.json) include parameters, input hashes and software versions. On the public pair, 75.34% of GT-valid pixels were accepted; accepted-point median 3D error was 0.74 cm and the 95th percentile was 11.90 cm. Thus the baseline misses the proposed 95% within 10 cm criterion. The scene's depth 5th–95th percentile was 2.22–4.66 m, so it also does not validate the requested 5–20 m range.

Run synthetic geometry without downloading data or starting services:

```bash
.venv/bin/python -m unittest discover -s tests -p 'test_two_view_localization.py' -v
.venv/bin/python eval/two_view_localization.py --output /tmp/two-view-synthetic.json
```

For the public pair, follow the proposal's [reproduction instructions](../docs/two-view-localization.md#reproduce). All 10 geometry tests passed and an independent rerun reproduced the combined JSON exactly.

Gaussian fusion, learned multi-view/stereo inference, generative completion and surveyed GPS/AGL accuracy have **not** been evaluated. Future fusion evaluation must report both metric error before/after splat refinement and rendering quality, including held-out views where available. Splat shape covariance and opacity must not be reported as position uncertainty or confidence; generated observations must retain synthetic provenance.
