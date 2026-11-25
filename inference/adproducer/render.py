"""Step 4: prompt + reference images -> the ad (MiniMax H3, Ref2VA mode).

H3 runs as an sglang server (`scripts/serve_h3.sh`, H3_URLS).  Same settings as our benchmark: 480p short edge,
16:9, 10 s, 50 steps, seed 1101; the audio track is removed (silent ad).
"""
from __future__ import annotations

from pathlib import Path


async def render(prediction: dict, run_dir: Path, out_mp4: Path) -> Path:
    from .core import adproducer_rl as adp
    refs = [Path(run_dir) / r["local_path"] for r in prediction["reference_images"] if r.get("local_path")]
    out_mp4.parent.mkdir(parents=True, exist_ok=True)
    await adp.H3Client().generate(prediction["gen_prompt"], refs, out_mp4)
    return out_mp4
