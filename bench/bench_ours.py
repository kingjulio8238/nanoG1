"""nanoG1 — engine throughput benchmark (the SHIPPED g1gpu engine, reproducible).

    modal run bench/bench_ours.py                 # production config (what trains the policy)
    modal run bench/bench_ours.py --config matched # warp-matched solver (dt 0.002, Newton 3/5)

Builds the EXACT same engine train.py uses — the pinned PufferLib fork (recipe.FORK_PIN),
compiled with recipe.TASK_FLAGS — then runs the fork's own `profile envspeed`:
it creates the g1gpu vector-env from config/g1gpu.ini, steps it with an empty-net
callback (NO learner, NO inference), and reports environment-step throughput. We
multiply by the decimation to get physics steps/s — directly comparable to
mujoco_warp / MJX raw stepping (bench_warp.py / bench_mjx.py).

Everything here comes from the pinned fork + tools/extract_g1_model.py, so the
number reproduces from a clean clone (`bash setup.sh` builds the same engine).
Prints a JSON blob between === ULTRA-BENCH RESULT === markers.
"""
import json, os, re, subprocess, sys, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # repo root for recipe
import modal
import recipe as R

GPU  = os.environ.get("NANOG1_GPU", "RTX-PRO-6000")
ARCH = {"RTX-PRO-6000": "sm_120", "H100": "sm_90", "L40S": "sm_89", "A100": "sm_80"}[GPU]
CUDA = "12.8.1" if ARCH in ("sm_120", "sm_100") else "12.6.3"
PUFFER, MODEL = "/root/PufferLib", "/root/envs/g1/model/g1.mjb"

# build-flag sets. production = recipe (what trains the policy). matched = warp's
# solver settings, for the apples-to-apples vs mujoco_warp / MJX line.
CONFIGS = {
    "production": R.TASK_FLAGS,
    "matched":    "-DG1_DT=0.002f -DENV_DECIMATION=5 -DSOL_ITER=3 -DSOL_LS_ITER=5 -DG1_TASK_V3 -DG1_PD_UNITREE",
}
DECIMATION = 5
BATCHES = [4096, 8192, 16384, 32768]
HORIZON = 512

app = modal.App("nanoG1-bench")

image = (
    modal.Image.from_registry(f"nvidia/cuda:{CUDA}-cudnn-devel-ubuntu22.04", add_python="3.11")
    .env({"PYTHONUNBUFFERED": "1", "DEBIAN_FRONTEND": "noninteractive"})
    .apt_install("git", "curl", "clang", "ccache", "libomp-dev",
                 # raylib (desktop) is linked into the --profile binary
                 "libgl1-mesa-dev", "libx11-dev", "libxrandr-dev", "libxinerama-dev",
                 "libxcursor-dev", "libxi-dev", "libxext-dev")
    .pip_install("torch", "mujoco==3.9.0", "playground==0.2.0", "jax", "numpy",
                 "pybind11", "setuptools", "rich", "rich_argparse", "gpytorch",
                 "scikit-learn", "wandb")
    .add_local_file("tools/extract_g1_model.py", "/root/extract_g1_model.py", copy=True)
    .run_commands(
        f"git clone -b {R.FORK_BRANCH} {R.FORK} {PUFFER} && cd {PUFFER} && "
        f"git checkout {R.FORK_PIN} && git log --oneline -1",
        f"pip install -e {PUFFER} --no-deps",
        "G1_MODEL_DIR=/root/envs/g1/model python /root/extract_g1_model.py",
    )
    .add_local_python_source("recipe")   # the remote fn imports recipe at runtime
)


@app.function(image=image, gpu=GPU, cpu=8.0, memory=32 * 1024, timeout=3600)
def bench(config: str = "production"):
    t0 = time.perf_counter()
    flags = CONFIGS[config]
    os.chdir(PUFFER)
    os.environ["G1_MODEL_PATH"] = MODEL
    print(subprocess.run(["git", "log", "--oneline", "-1"], capture_output=True, text=True).stdout.strip(), flush=True)

    # build.sh compiles the env's *_gpu.cu (defining my_gpu_*) only in DEFAULT mode and
    # archives it into the static lib; --profile skips that. So build the env first,
    # THEN the profile binary (which links the now-complete static lib).
    env = f"NVCC_ARCH={ARCH} G1_TASK_FLAGS='{flags}'"
    for step in ("./build.sh g1gpu", "./build.sh g1gpu --profile"):
        print(f"[{config}] {step}   ({flags})", flush=True)
        b = subprocess.run(f"{env} {step}", shell=True, capture_output=True, text=True)
        if b.returncode != 0:
            print(b.stdout[-2500:]); print(b.stderr[-4000:])
            raise SystemExit(f"build failed ({config}): {step}")

    # sweep batch sizes; envspeed prints "throughput: X M steps/s" (env/control steps)
    rows = []
    for ta in BATCHES:
        out = subprocess.run(
            f"G1_MODEL_PATH={MODEL} ./profile envspeed --total-agents {ta} --horizon {HORIZON}",
            shell=True, capture_output=True, text=True).stdout
        m = re.search(r"throughput:\s*([\d.]+)\s*M steps/s", out)
        env_sps = float(m.group(1)) * 1e6 if m else None
        rows.append({"total_agents": ta,
                     "env_steps_per_s": round(env_sps, 1) if env_sps else None,
                     "physics_steps_per_s": round(env_sps * DECIMATION, 1) if env_sps else None})
        print(f"  total_agents={ta:>6}  env={env_sps and round(env_sps/1e6,3)}M/s  "
              f"physics={env_sps and round(env_sps*DECIMATION/1e6,3)}M/s", flush=True)

    valid = [r for r in rows if r["physics_steps_per_s"]]
    peak = max(valid, key=lambda r: r["physics_steps_per_s"]) if valid else None

    print("\n=== ULTRA-BENCH RESULT ===")
    print(json.dumps({
        "engine": "nanoG1 g1gpu (shipped fork @ " + R.FORK_PIN + ")",
        "metric": "environment-step throughput, no learner (profile envspeed) x decimation",
        "config": config, "task_flags": flags, "decimation": DECIMATION,
        "gpu": GPU, "horizon": HORIZON, "sweep": rows,
        "peak_physics_steps_per_s": peak["physics_steps_per_s"] if peak else None,
        "peak_env_steps_per_s": peak["env_steps_per_s"] if peak else None,
        "peak_at_total_agents": peak["total_agents"] if peak else None,
        "wall_s": round(time.perf_counter() - t0, 1),
    }, indent=2))
    print("=== END RESULT ===\n", flush=True)
    return peak


@app.local_entrypoint()
def main(config: str = "production"):
    r = bench.remote(config=config)
    if r:
        print(f"\n✓ {config}: peak {r['physics_steps_per_s']/1e6:.2f}M physics steps/s "
              f"({r['env_steps_per_s']/1e6:.2f}M env steps/s @ {r['total_agents']} agents) on {GPU}")
