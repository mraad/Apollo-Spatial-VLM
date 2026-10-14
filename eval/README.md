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

## Recommendations

1. **VLM:** switch the demo to Qwen3.6-27B (smallest change: send `enable_thinking: false`). Or adopt each model's native output format, which also fixes Qwen3-VL-32B. For speed, Molmo2-8B, with its own prompt and parser path.
2. **Depth:** offer MoGe-3 ViT-L as the default on Linux/CUDA. It has better metric depth, a field-of-view estimate instead of the 60° guess, normals for splat orientation, and an MIT license, for about 160 ms per image. Keep DA3 on macOS, where MoGe-3 doesn't run (it needs Triton), and as the fast path.
3. Use EXIF focal length whenever it's present, whichever model runs.

## Caveats

- One depth dataset (DIODE), 200 RefCOCO samples, one run each. vLLM isn't fully deterministic even at temperature 0.
- RefCOCO asks for one referred object; the demo usually asks for several targets per prompt.
- Latency is in eager mode (as `run-vlm.sh` runs it), one request at a time, on Blackwell GPUs.
- The native Qwen runs were re-scored from their stored raw answers after the parser learned the `"bbox"` key Qwen3-VL-32B uses. This is identical to re-running with the fixed parser.

## Reproduce (RTX PRO 6000 server)

```bash
uv run eval/prepare_refcoco.py                    # 200 RefCOCO refs + their COCO images -> eval/data/
curl -O http://diode-dataset.s3.amazonaws.com/val.tar.gz && tar xzf val.tar.gz -C eval/data   # DIODE val
eval/serve.sh 0 8012 qwen3-vl-32b <Qwen3-VL-32B path or id>
eval/serve.sh 1 8014 qwen3.6-27b Qwen/Qwen3.6-27B
eval/serve.sh 2 8013 molmo2-8b allenai/Molmo2-8B --max-num-batched-tokens 16384
.venv/bin/python eval/vlm_eval.py --url http://127.0.0.1:8014/v1 --model qwen3.6-27b --name qwen3.6-27b --no-think [--prompt native] [--max-dim 1024]
.venv/bin/python eval/vlm_eval.py --url http://127.0.0.1:8013/v1 --model molmo2-8b --name molmo2-8b --kind molmo
CUDA_VISIBLE_DEVICES=3 .venv/bin/python eval/depth_eval.py --model da3        # or da3-gt-focal
uv venv --python 3.12 eval/.venv-moge && uv pip install --python eval/.venv-moge --torch-backend auto torch torchvision git+https://github.com/microsoft/MoGe.git
CUDA_VISIBLE_DEVICES=3 eval/.venv-moge/bin/python eval/depth_eval.py --model moge3-l   # or moge3-g
```
