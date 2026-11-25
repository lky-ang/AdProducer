# AdCreation-Bench evaluation

**AdCreation-Bench** holds 305 held-out ad briefs: each names a real product, food, place or vehicle only
indirectly and lists the details that must be correct. Every object is unique and appears in neither the SFT nor
the RL data. The briefs cover 8 domains (consumer electronics 58, travel 48, fashion 45, food 43, beauty 35,
automotive 34, home 29, anime/IP merch 13) and 3 freshness levels (recent 122, long-tail 130, evergreen 53).

Data: [AdProducer/AdCreation-Bench](https://huggingface.co/datasets/AdProducer/AdCreation-Bench): `AdCreation-Bench.json`
(brief, meta, verified facts, official images, text requirement, GT paths), `official_images.zip`, `GT-Video/`
(`gt_video.zip` with the MiniMax H3 ground-truth ads, plus their inputs `gen_prompt.json` and `images.zip`) and
`eval_prompts/` (the judge prompts).

AdCreation-Bench evaluates advertising quality from two complementary perspectives: **PG-Score** (product fidelity,
with ground truth) and **AD-Score** (ad effectiveness, without ground truth).

## PG-Score (Product-Grounding Score)

The method under test turns the brief into a MiniMax H3 input (agents: prompt + chosen reference images), H3 renders
it (seed 1101, 480p, 16:9, 10 s), and Gemini 3.1 Pro (temperature 0) judges the video against the brief, the verified
facts, the official images and the GT video:

```
PG-Score = 0.4 visual_correctness + 0.2 text_accuracy + 0.2 selling_point   (each 0 / .25 / .5 / .75 / 1)
         + 0.1 shot_design + 0.1 aesthetics                                 (each 0 / .5 / 1)
```

When the brief needs no readable text, `text_accuracy` is dropped and its weight is spread over the rest. As in
Gen-Searcher (K-Score), the benchmark metric is the RL video reward: same judge prompt
(`inference/adproducer/core/prompts/reward_video.txt`) and inputs.

## AD-Score

An independent GPT judge (we use GPT-6-sol) watches the ad as frames (2 fps, shots detected with PySceneDetect) next
to the brief, the verified facts and the first three official images, without any ground truth, and rates it as a
commercial: brief_fidelity, product_presentation, visual_quality, narrative_flow, hook, creative_quality,
persuasiveness and purchase_intent (each 1-5), plus overall_ad_quality (1-10). Prompt: `prompts/ad_judge.txt`;
script: `ad_score.py`.

## Environment

The same environment as [inference](../inference/README.md), plus the AD-Score dependencies:

```bash
conda activate adproducer
pip install -r benchmark/requirements.txt          # scenedetect + OpenCV for AD-Score; ffmpeg on PATH
```

Keys and endpoints come from `inference/.env` (search tools, policy and MiniMax H3 servers). Both judges read the
`JUDGE_*` variables, so set them for each command: Gemini 3.1 Pro for `bench_eval.py score` (`JUDGE_BACKEND`,
`JUDGE_MODEL`, `JUDGE_API_KEY`, as in `training/rl/.env.example`), and an OpenAI-compatible endpoint serving the GPT
judge for `ad_score.py` (`JUDGE_BASE_URL`, `JUDGE_API_KEY`, `JUDGE_MODEL`).

## Data

```bash
hf download AdProducer/AdCreation-Bench --repo-type dataset --local-dir data/AdCreation-Bench
(cd data/AdCreation-Bench && unzip -q official_images.zip && unzip -q GT-Video/gt_video.zip)   # -> official_images/, gt_video/
```

## Run

`bench_eval.py` imports the evaluated agent code from `inference/adproducer/core/` (loop, tools, H3 client, judge).

```bash
B=data/AdCreation-Bench; E="python benchmark/bench_eval.py"

$E infer    --bench-dir $B --out-dir runs/adproducer --base-url http://127.0.0.1:8000/v1 --model adproducer
$E generate --bench-dir $B --out-dir runs/adproducer                       # H3 Ref2VA videos
$E score    --bench-dir $B --out-dir runs/adproducer                       # PG-Score per item
$E summary  --bench-dir $B --out-dir runs/adproducer                       # means by domain / freshness
python benchmark/ad_score.py --bench-dir $B --videos runs/adproducer/videos --out runs/adproducer/ad_score.jsonl

$E gen-gt   --bench-dir $B                                                  # optional: re-render the GT (unzip GT-Video/images.zip -d GT-Video first)
```

`infer --baseline brief` evaluates the "brief straight into H3" baseline. Agents run at temperature 0.6, top-p 0.9,
one run per item, at most 8 tool calls, 6144 new tokens per turn; failures are skipped and counted, as in
Gen-Searcher's `gpt_eval_knowgen.py`.

To score videos made by another system, put them in `runs/<method>/videos/<id>.mp4` and run `score` (and `ad_score.py`)
directly; without `predictions.jsonl` the judge is told that the planned shot times are not available.
