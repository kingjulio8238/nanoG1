// Thin C shim exposing the *exact* nanoG1 policy forward pass to Python (ctypes).
//
// We reuse PufferLib's puffernet.h — the same inference code path the browser demo
// and eval.py use — so on-robot inference is bit-identical to what we validated.
// Build with deploy/build_policy.sh (needs the engine fork from `bash setup.sh`).
#include <stdlib.h>
#include "puffernet.h"

// nanoG1 (v3) policy shape: obs 98 -> MLP(128, 3 layers) -> 29 joint means.
#define NANOG1_OBS 98
#define NANOG1_NU  29

static PufferNet* g_net = NULL;

int  nn_obs(void) { return NANOG1_OBS; }
int  nn_nu(void)  { return NANOG1_NU;  }

// Load assets/nanoG1.bin and build the net. Returns 0 on success, <0 on failure.
int nn_init(const char* path) {
    Weights* w = load_weights(path);
    if (!w) return -1;
    int ls[NANOG1_NU]; for (int i = 0; i < NANOG1_NU; i++) ls[i] = 1;
    g_net = make_puffernet(w, 1, NANOG1_OBS, 128, 3, ls, NANOG1_NU);
    return g_net ? 0 : -2;
}

// obs: float[98] in, act: float[29] out (raw policy means, before clip/scale).
void nn_infer(const float* obs, float* act) {
    if (!g_net) { for (int i = 0; i < NANOG1_NU; i++) act[i] = 0.0f; return; }
    forward_puffernet(g_net, (float*)obs, act);
}
