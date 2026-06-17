# nanoG1

**Train a [Unitree G1](https://www.unitree.com/g1) humanoid to walk in under 60 seconds, on a single GPU — pure RL, from scratch.**

No demonstrations, no reference gait, no motion capture. The policy starts from noise and learns to walk from reward alone in **~59 seconds** of wall-clock training (~75M samples at 1.28M samples/s) for about **$0.17** on one GPU.

🤖 **[Live demo — drive the trained G1 in your browser](https://nanog1.com)** &nbsp;·&nbsp; 🤗 **[Model on Hugging Face](https://huggingface.co/kingJulio/nanoG1)**

![nanoG1](assets/preview.png)

This is to robot locomotion what [nanoGPT](https://github.com/karpathy/nanoGPT) is to language models: the smallest, most legible thing that actually works, that you can read top-to-bottom and run yourself.

---

## Quickstart

```bash
git clone https://github.com/kingjulio8238/nanoG1 && cd nanoG1
bash speedrun.sh
```

That's it. `speedrun.sh` syncs the Python env, fetches the engine, trains the G1 on a GPU (via [Modal](https://modal.com)), gates the result, and drops the trained policy at `assets/nanoG1.bin`.

**Prereqs:** [`uv`](https://docs.astral.sh/uv), a [Modal](https://modal.com) account (`modal token new`), and `git`. The GPU run is the only paid part (~$0.17 on an RTX PRO 6000); everything else is local and free.

Want to turn the dials yourself instead of one-shotting it:

```bash
bash setup.sh                          # fetch the G1-specialized engine (pinned fork)
modal run train.py --smoke             # ~$0.02 — validate the whole stack first
modal run train.py                     # the <60s walk -> assets/nanoG1.bin
python eval.py assets/nanoG1.bin       # quality gate: does it actually walk?
bash web/build_demo.sh && ./build/g1demo assets/nanoG1.bin   # watch it locally
```

Train on a different card: `NANOG1_GPU=H100 modal run train.py`.

### Run it on a real robot

Put the policy on a physical **Unitree G1**:

```bash
bash setup.sh                 # engine fork (for puffernet.h), once
bash deploy/build_policy.sh   # build the inference shim
python deploy/deploy_g1.py --net eth0          # walk in place
python deploy/deploy_g1.py --net eth0 --teleop # WASD drive
```

It runs a 50 Hz loop over Unitree's low-level DDS interface (`unitree_sdk2py`):
robot state → the exact trained observation → policy → joint PD targets, with a
zero-torque → move-to-home → policy safety sequence. **The policy is sim-trained —
hang the robot from a gantry and keep the E-stop in hand.** Full guide and the
hardware checklist: [`deploy/README.md`](deploy/README.md).

---

## What you get

| | |
|---|---|
| **Time-to-walk** | **58.9 s** (75M samples @ 1.28M SPS, single RTX PRO 6000) |
| **Cost-to-walk** | **~$0.17** |
| **Method** | PPO + V-trace, **pure RL from scratch** — no demos, no reference motion |
| **Physics** | MuJoCo-grade soft-convex contact, friction cones, domain randomization |
| **Engine throughput** | **8.5M physics steps/s** (production) · **1.8× mujoco_warp** at matched settings |

### Engine throughput — G1, RTX PRO 6000, physics steps/s (identical settings)

```
nanoG1        ████████████████████████████████████  7.25M
mujoco_warp   ████████████████████                   4.0M
Genesis*      ███████████                            2.3M
MJX           █████▌                                 1.1M
```

Apples-to-apples at **identical settings** (dt 0.002, Newton 3/5): **1.8× mujoco_warp**.
In its own production config (dt 0.004, Newton 2/3 — what trains the policy), nanoG1's
engine runs at **8.5M**. Reproduce both from a clean clone: `modal run bench/bench_nanog1.py`.

\* Genesis runs its own (non-MuJoCo) solver — a competitor datapoint, not matched-physics. See [RESULTS.md](RESULTS.md) for exact settings, env-step throughput, and provenance.

---

## How it works

The thesis: MuJoCo's physics isn't inherently slow for RL — it's just never been **specialized**. nanoG1 compiles the simulator *per-robot*. For a fixed G1, the kinematic tree, contact set, and solver layout are compile-time constants, so the whole step inlines into straight-line CUDA with no runtime dispatch, no broadphase, and a fixed-iteration solver. That's where the throughput comes from — not from cheapening the physics (it's validated trajectory-by-trajectory against the MuJoCo C engine).

Two ingredients make it learn to walk this fast:

1. **A G1-specialized GPU engine** — a [pinned PufferLib fork](https://github.com/kingjulio8238/PufferLib/tree/g1) that bakes the G1 in at compile time (zero Python in the hot loop). `recipe.py` pins the exact commit.
2. **A left↔right symmetry loss** (N1, after [Yu et al. 2018](https://arxiv.org/abs/1801.08093)) — regularizing the policy toward a mirror-symmetric gait cut samples-to-walk ~26% *and* smoothed the gait. That's the single biggest lever.

Everything else — the reward weights, PPO/Muon hyperparameters, the dt/decimation/solver settings — lives in **one file, [`recipe.py`](recipe.py)**. That's the dial you turn.

---

## Repo layout

```
recipe.py        the frozen winning recipe — the one dial you turn
train.py         Modal launcher: builds the engine, trains, pulls the walk checkpoint
eval.py          quality gate — runs the host-physics battery, checks it walks
speedrun.sh      one command: env -> engine -> train -> gate
setup.sh         fetch the pinned G1 engine (for local demo/eval builds)
web/             browser demo (raylib + the policy, host physics) -> WASM
deploy/          run the policy on a REAL Unitree G1 (unitree_sdk2py, low-level DDS)
bench/           competitor benchmarks (warp / MJX / Genesis) — same card, same G1
tools/           bake the G1 model + meshes from MuJoCo (assets are committed)
assets/nanoG1.bin   the trained <60s policy (655 KB)
```

---

## Credits

**nanoG1's engine is [PufferLib](https://github.com/PufferAI/PufferLib).** The
whole approach — compile-time per-environment specialization, zero Python in the
hot loop, the CUDA trainer, the [Muon](https://github.com/PufferAI/PufferLib)
optimizer path, PufferNet — is PufferLib's, and the G1 simulator is built as a
PufferLib environment. nanoG1 would not exist without it. Huge thanks to
[@jsuarez5341](https://github.com/jsuarez5341) and the PufferLib contributors.
PufferLib is MIT-licensed; we carry its license forward.

Also built on [MuJoCo](https://github.com/google-deepmind/mujoco) physics
semantics, the [Unitree G1](https://github.com/google-deepmind/mujoco_menagerie)
from MuJoCo Menagerie, and [raylib](https://github.com/raysan5/raylib) for the
demo. Compute on [Modal](https://modal.com). Inspired by
[nanoGPT](https://github.com/karpathy/nanoGPT) and
[nanochat](https://github.com/karpathy/nanochat).

MIT licensed.
