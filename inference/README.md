# Inference: product link → ad

## How it works

| step | what happens | code |
|---|---|---|
| 1. Product link | read the product page: title, bullets, specification table | `adproducer/product_page.py` |
| 2. Brief | an LLM turns the page into a benchmark-style brief: the product named by description, one selling point, and the details that must be correct, **without their values** | `adproducer/brief.py`, `adproducer/prompts/brief_writer.txt` |
| 3. Search agent | the AdProducer policy (Qwen3.5-9B, SFT + RL) thinks, calls one tool per round (`search`, `image_search`, `browse`; at most 8 calls, at most 2 image searches) and answers with a multi-shot MiniMax H3 prompt that cites the reference images it chose | `adproducer/agent.py`, `adproducer/core/adproducer_rl.py` |
| 4. Ad | MiniMax H3 in reference-to-video (Ref2VA) mode renders the prompt with the chosen images: 10 s, 16:9, 480p (H3 also generates audio; this pipeline keeps the video track only) | `adproducer/render.py` |

Steps 1-2 are a convenience for starting from a product link: they write the kind of brief AdProducer was trained
on. They are not part of training or of the AdCreation-Bench protocol, where the brief is given. The agent loop, tool
budget, answer checks and H3 settings in `adproducer/core/` are the ones used to evaluate the policy on
AdCreation-Bench, so a production run behaves like the benchmark.

## Environment

```bash
conda create -n adproducer python=3.12 -y && conda activate adproducer
pip install -r requirements.txt
pip install "sglang[all]==0.5.23"           # serves the policy (tested version); ffmpeg must be on PATH
cp .env.example .env                        # fill in the keys / endpoints below
```

| what | used in | where to get it / variables |
|---|---|---|
| Policy checkpoint | step 3 | [AdProducer-RL-9B](https://huggingface.co/AdProducer/AdProducer-RL-9B) (RL) or [AdProducer-SFT-9B](https://huggingface.co/AdProducer/AdProducer-SFT-9B) (SFT only); served on one GPU by `scripts/serve_policy.sh` |
| MiniMax H3 | step 4 | weights and sglang build from the MiniMax-H3 repository; Ref2VA on two GPUs via `scripts/serve_h3.sh`; `H3_URLS` |
| Search tools | step 3 | [Serper](https://serper.dev) key for `search` / `image_search` (`SERPER_KEY_ID`), [Jina Reader](https://jina.ai/reader) key for `browse` (`JINA_API_KEYS`), any OpenAI-compatible model to summarise browsed pages (`BROWSE_SUMMARY_*`) |
| Brief writer | step 2 | any OpenAI-compatible chat model (`BRIEF_*`) |

## Run

```bash
set -a; . ./.env; set +a
scripts/serve_policy.sh AdProducer/AdProducer-RL-9B 0                 # policy on GPU 0, :8000
python -m adproducer brief "https://www.amazon.com/dp/B0DP64X2LT" --out out/heater   # steps 1-2
python -m adproducer agent --out out/heater                                           # step 3
# stop the policy server if the GPUs are shared, then MiniMax H3 on two GPUs, :30011
scripts/serve_h3.sh /path/to/MiniMax-H3
python -m adproducer render --out out/heater                                          # step 4 -> out/heater/ad.mp4
```

`python -m adproducer run <url> --out <dir>` runs all four steps when both servers are up. If a shop blocks
scripted requests, save the page in a browser and pass `--html page.html`; `--brief-text` skips step 2.

## Example: Amazon Basics space heater (zero-shot)

`examples/amazon_heater/` holds one complete run on
[an Amazon product page](https://www.amazon.com/dp/B0DP64X2LT): the parsed page (`product.json`), the brief
(`brief.txt`), the full agent trajectory (`trajectory.json`), the final prompt with its reference images
(`prediction.json`) and the delivered ad (`ad.mp4`).

In this run the agent made two web searches, two image searches and one page read, then answered with a three-shot
prompt and four reference images.
