"""Runtime patches to slime@540ea91 (+ the slimerl/slime:latest Megatron) needed by AdProducer RL.

Called by train_rl.sh after cd $SLIME_DIR.  Idempotent; exits non-zero if a target is not found, so a
changed slime version fails loudly instead of silently training without the fix.

1. mrope + context parallelism (correctness).  slime's Qwen3_5MultimodalRotaryEmbedding skips the CP slice of the
   rotary table only when called with packed_seq=True, which the Megatron patch of slime v0.5.12 passed from GPTModel
   but docker/patch/latest no longer does.  Without it the table is CP-sliced as if unpacked and thd RoPE gets wrong
   positions (or a size mismatch).  Training is always packed here (the plugin rejects unpacked input).
2. Temperature scaling of the training logits (rollout T=0.7): `logits / T` made a second full fp32 [T, V/TP] copy in
   forward and a full-size gradient in backward (out of memory on long trajectories).  The temperature is now
   applied to each row chunk inside the chunked log-prob function (forward and recomputed backward), so no full-size
   tensor is divided.  Falls back to the original division when --log-probs-chunk-size is not set.
3. _ChunkedLogProbEntropy.backward allocated a full fp32 gradient next to the saved fp32 logits (out of memory).  Each
   row chunk of the saved logits is read exactly once, right before its gradient is computed, so the gradient is
   written back into the logits storage instead (same values, no extra allocation).
"""
import sys
from pathlib import Path

root = Path(sys.argv[1] if len(sys.argv) > 1 else ".")
PATCHES = [
    ("slime_plugins/models/qwen3_5_vl.py",
     "        packed_seq: bool = False,\n",
     "        packed_seq: bool = True,   # [adproducer patch] packed (thd) is the only training layout\n"),
    # 2. temperature: no full-size `logits / T` (forward copy + a full-size DivBackward gradient); applied per chunk
    ("slime/backends/megatron_utils/loss.py",
     "    if rollout_temperature != 1.0:\n        logits = logits / rollout_temperature\n    logits = logits.contiguous()\n",
     "    # [adproducer patch] temperature is applied per row chunk inside calculate_log_probs_and_entropy\n"
     "    chunk_temperature = rollout_temperature if args.log_probs_chunk_size > 0 else 1.0\n"
     "    if rollout_temperature != 1.0 and chunk_temperature == 1.0:\n"
     "        logits = logits / rollout_temperature\n"
     "    logits = logits.contiguous()\n"),
    ("slime/backends/megatron_utils/loss.py",
     "        log_prob_keep_mask=top_p_keep_mask,\n    )\n    log_prob_full = log_prob_full.squeeze(-1)",
     "        log_prob_keep_mask=top_p_keep_mask,\n        temperature=chunk_temperature,   # [adproducer patch]\n"
     "    )\n    log_prob_full = log_prob_full.squeeze(-1)"),
    ("slime/utils/ppo_utils.py",
     "    with_entropy_grad: bool = True,\n):\n    logits = logits.contiguous()\n    entropy = None\n",
     "    with_entropy_grad: bool = True,\n    temperature: float = 1.0,   # [adproducer patch] chunked path only\n"
     "):\n    logits = logits.contiguous()\n    entropy = None\n"),
    ("slime/utils/ppo_utils.py",
     "                logits, tokens, log_prob_keep_mask, tp_group, with_entropy, with_entropy_grad, chunk_size\n            )",
     "                logits, tokens, log_prob_keep_mask, tp_group, with_entropy, with_entropy_grad, chunk_size,\n"
     "                temperature,   # [adproducer patch]\n            )"),
    ("slime/utils/ppo_utils.py",
     "    def forward(ctx, logits, tokens, keep_mask, tp_group, with_entropy, with_entropy_grad, chunk_size):\n",
     "    def forward(ctx, logits, tokens, keep_mask, tp_group, with_entropy, with_entropy_grad, chunk_size,\n"
     "                temperature=1.0):   # [adproducer patch]\n"),
    ("slime/utils/ppo_utils.py",
     "            values, entropies = _calculate_log_probs_and_entropy_chunk(\n                logits[start:end],\n",
     "            values, entropies = _calculate_log_probs_and_entropy_chunk(\n"
     "                logits[start:end] if temperature == 1.0 else logits[start:end] / temperature,   # [adproducer patch]\n"),
    ("slime/utils/ppo_utils.py",
     "        ctx.with_entropy_grad = with_entropy and with_entropy_grad\n        ctx.chunk_size = chunk_size\n",
     "        ctx.with_entropy_grad = with_entropy and with_entropy_grad\n        ctx.chunk_size = chunk_size\n"
     "        ctx.temperature = temperature   # [adproducer patch]\n"),
    # 3. the gradient goes into the saved logits' storage (each row chunk is read once, right before its gradient)
    ("slime/utils/ppo_utils.py",
     "        logits, tokens, keep_mask = ctx.saved_tensors\n        grad_input = torch.empty_like(logits)\n",
     "        logits, tokens, keep_mask = ctx.saved_tensors\n"
     "        grad_input = logits   # [adproducer patch] reuse the saved logits' storage (each chunk read once)\n"),
    ("slime/utils/ppo_utils.py",
     "                values, entropies = _calculate_log_probs_and_entropy_chunk(\n                    chunk,\n",
     "                values, entropies = _calculate_log_probs_and_entropy_chunk(\n"
     "                    chunk if ctx.temperature == 1.0 else chunk / ctx.temperature,   # [adproducer patch]\n"),
    ("slime/utils/ppo_utils.py",
     "        return grad_input, None, None, None, None, None, None\n",
     "        return grad_input, None, None, None, None, None, None, None   # [adproducer patch] + temperature\n"),
]
ok = True
for rel, old, new in PATCHES:
    p = root / rel
    s = p.read_text()
    if new in s:
        print(f"[patch_slime] already applied: {rel}: {new.splitlines()[0].strip()[:60]}")
        continue
    if s.count(old) != 1:
        print(f"[patch_slime] FAILED: target not found exactly once in {rel}: {old.splitlines()[0].strip()[:60]}")
        ok = False
        continue
    p.write_text(s.replace(old, new))
    print(f"[patch_slime] applied: {rel}: {new.splitlines()[0].strip()[:60]}")
sys.exit(0 if ok else 1)
