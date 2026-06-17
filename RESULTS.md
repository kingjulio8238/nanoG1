# Results

Curated, honest numbers with provenance. The rule from day one: identical physics
settings across every engine compared, three metrics always reported together, and
compile/JIT time separated from steady-state throughput.

## Time-to-walk

| metric | value |
|---|---|
| **time-to-walk** | **58.9 s** |
| samples-to-walk | 75M control steps |
| steady SPS | 1.28M samples/s (end-to-end: env + inference + learning) |
| cost-to-walk | ~$0.17 |
| GPU | 1× RTX PRO 6000 (sm_120) |
| method | PPO + V-trace + Muon, **pure RL from scratch** (no demos, no reference gait) |
| seed | 42 |

**What "time-to-walk" means.** The policy is trained on a 150M-step schedule (the LR
anneals over the full budget). It crosses the frozen quality gate at ~75M samples;
`time-to-walk = samples-to-walk / steady-SPS = 75M / 1.28M ≈ 59 s`. `train.py`
captures the checkpoint nearest 75M and ships it as `assets/nanoG1.bin`. Training is
deterministic per built binary (fixed seed + pinned engine commit).

## Quality gate (the frozen bar)

`eval.py` runs the MuJoCo-validated host-physics battery (the same stepper the
browser demo uses) and checks all six thresholds. Frozen against a reference 116M
checkpoint, approved 2026-06-15:

| check | threshold | meaning |
|---|---|---|
| `battery_falls` | ≤ 1 | falls across the command battery |
| `battery_perf` | ≥ 0.90 | velocity-tracking score |
| `action_jerk_rms` | ≤ 0.21 | action smoothness |
| `ang_vel_xy_rms` | ≤ 0.21 | torso wobble |
| `yaw_rate_rms` | ≤ 0.20 | heading stability |
| `leg_qvel_rms` | ≤ 1.22 | leg-velocity smoothness |

"Passes the gate" is re-provable by one command (`python eval.py assets/nanoG1.bin`),
not by testimony.

## Engine throughput — the wall

G1, single **RTX PRO 6000**, physics steps/s. nanoG1's number is the **shipped
engine** — the pinned fork that `setup.sh`/`train.py` build — measured by the fork's
own `profile envspeed`: environment stepping with **no learner**, × decimation 5.
Reproduce it from a clean clone with `modal run bench/bench_nanog1.py`. The
MuJoCo-physics engines load the **same** G1 model (md5 `432c765a`) — the
apples-to-apples guarantee.

| engine | physics steps/s | env steps/s | settings | note |
|---|---|---|---|---|
| **nanoG1 (production)** | **8.5M** | 1.70M | dt 0.004, Newton 2/3 | the config the shipped policy trains with |
| nanoG1 (matched) | 7.25M | 1.45M | dt 0.002, Newton 3/5 | warp's exact settings → **1.8× warp** |
| mujoco_warp | 4.0M | — | dt 0.002, Newton 3/5 | needs `--nconmax 32 --njmax 128` (per-world capacities; G1 nefc≈85) |
| Genesis\* | 2.28M | — | its own solver | \*different physics — see caveat |
| MJX | 1.12M | — | dt 0.002, Newton 3/5 | jit(vmap(step)) repeated-step (not lax.scan) |

**Honesty notes.**
- The rigorous, matched-physics claim is **1.8× mujoco_warp** (7.25M vs 4.0M at
  identical dt/solver). The **8.5M** headline is nanoG1's *production* config
  (dt 0.004, Newton 2/3) — what the shipped policy actually trains under. Both
  reproduce from `bench_nanog1.py`; don't conflate them.
- **Scope:** these are stepping-only numbers (no learner). End-to-end during
  *training* (env + inference + learning) the engine sustains ~1.28M SPS =
  **~6.4M physics-equiv steps/s** — what `train.py` reports. 8.5M is the raw
  stepping ceiling; 6.4M is what you see while actually training.
- **Per-GPU:** throughput scales with the card — substantially lower on smaller
  GPUs (a community fork measured ~2.5M physics-equiv on a 3090). Always report the GPU.
- **Genesis** runs its own (non-MuJoCo) solver + contact model and reparses the
  MJCF — a *competitor* datapoint, not matched physics; `bench_genesis.py` also
  reports the dt-normalized `sim_s_per_wall_s`. Quote with the caveat.
- warp / MJX / nanoG1 are **bit-comparable** (same model fingerprint). Genesis is not.
- An earlier draft cited 8.9M from a standalone research kernel that wasn't shipped;
  this 8.5M is the **shipped** engine, reproducible from `setup.sh`.

## Reproduce

```bash
# nanoG1 engine — the shipped fork; reproduces the headline
NANOG1_GPU=RTX-PRO-6000 modal run bench/bench_nanog1.py                  # production
NANOG1_GPU=RTX-PRO-6000 modal run bench/bench_nanog1.py --config matched # vs warp (matched)

# training (writes the result blob with steady_sps, T_walk_s, est_cost_usd)
modal run train.py --smoke      # validate the stack first (~$0.02)
modal run train.py

# competitors (smoke each first), on the same card
NANOG1_GPU=RTX-PRO-6000 modal run bench/bench_warp.py --nconmax 32 --njmax 128
NANOG1_GPU=RTX-PRO-6000 modal run bench/bench_mjx.py
NANOG1_GPU=RTX-PRO-6000 modal run bench/bench_genesis.py
```

Each bench prints a JSON blob between `=== ULTRA-BENCH RESULT ===` markers carrying
the model fingerprint, the exact `opt` physics settings, and `run_meta`
(`total_wall_s`, `est_cost_usd`). The cost is an estimate from the `RATE_*`
constants — calibrate against your Modal dashboard after the first smoke.
