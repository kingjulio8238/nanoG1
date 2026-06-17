"""Run the nanoG1 walking policy on a REAL Unitree G1 (29-DoF), via the
unitree_sdk2py low-level (HG) interface over DDS.

    python deploy/deploy_g1.py --net eth0            # walk in place
    python deploy/deploy_g1.py --net eth0 --teleop   # WASD drive

╔════════════════════════════════════════════════════════════════════════════╗
║  SAFETY — READ deploy/README.md FIRST.  Hang the robot from a gantry / have  ║
║  the remote E-stop in hand. The policy was trained in sim; first hardware    ║
║  runs WILL be rough. Start suspended, low command, ready to kill power.      ║
╚════════════════════════════════════════════════════════════════════════════╝

The observation, action mask, gains, home pose and 50 Hz / 0.8 s gait phase here
are transcribed verbatim from the validated reference (web/g1_demo.c +
web/g1_model_const.h). Inference uses the exact PufferNet forward via
libnanog1policy (build with deploy/build_policy.sh). Joint order is the standard
29-DoF G1 order (legs 0-11, waist 12-14, arms 15-28) — the same order the policy
was trained in; VERIFY it matches your robot's motor indices before running.
"""
import argparse, ctypes, math, os, sys, time
import numpy as np

# ── policy interface (must match web/g1_demo.c + web/g1_model_const.h) ──────────
NU            = 29          # actuated joints
LEG_DOF       = 12          # v3 policy controls legs only; waist+arms held at home
CONTROL_DT    = 0.02        # 50 Hz  (G1_DT 0.004 × decimation 5)
PHASE_PERIOD  = 40          # gait-clock period in control steps  → 0.8 s
ACTION_SCALE  = 0.25
ANG_VEL_SCALE = 0.25
DOF_VEL_SCALE = 0.05

# home / default joint angles (hc_key_qpos[7:], radians)
HOME = np.array([
    -0.10, 0.0, 0.0, 0.30, -0.20, 0.0,      # left leg
    -0.10, 0.0, 0.0, 0.30, -0.20, 0.0,      # right leg
     0.0,  0.0, 0.0,                         # waist  (yaw, roll, pitch)
     0.20, 0.20, 0.0, 1.28, 0.0, 0.0, 0.0,   # left arm
     0.20,-0.20, 0.0, 1.28, 0.0, 0.0, 0.0,   # right arm
], dtype=np.float64)

# PD gains. Legs: the v3 "Unitree" gains (g1_staged_kernels.cuh / g1_host.c:292).
# Waist+arms: hold at home with the model's actuator gains (hc_act_gain0[12:]).
KP = np.array([
    100,100,100,150,40,40,  100,100,100,150,40,40,   # legs
    75,75,75,                                          # waist
    75,75,75,75,2,2,2,  75,75,75,75,2,2,2,             # arms
], dtype=np.float64)
KD = np.array([
    2,2,2,4,2,2,  2,2,2,4,2,2,                         # legs
    2,2,2,                                              # waist
    2,2,2,2,0.2,0.2,0.2,  2,2,2,2,0.2,0.2,0.2,         # arms
], dtype=np.float64)

# position-target limits (hc_act_ctrlrange), shape (29, 2)
CTRL_RANGE = np.array([
    (-2.5307,2.8798),(-0.5236,2.9671),(-2.7576,2.7576),(-0.087267,2.8798),(-0.87267,0.5236),(-0.2618,0.2618),
    (-2.5307,2.8798),(-0.5236,2.9671),(-2.7576,2.7576),(-0.087267,2.8798),(-0.87267,0.5236),(-0.2618,0.2618),
    (-2.618,2.618),(-0.52,0.52),(-0.52,0.52),
    (-3.0892,2.6704),(-1.5882,2.2515),(-2.618,2.618),(-1.0472,2.0944),(-1.97222,1.97222),(-1.61443,1.61443),(-1.61443,1.61443),
    (-3.0892,2.6704),(-2.2515,1.5882),(-2.618,2.618),(-1.0472,2.0944),(-1.97222,1.97222),(-1.61443,1.61443),(-1.61443,1.61443),
], dtype=np.float64)

# command teleop step sizes (vx forward, vy lateral, wyaw turn) — kept conservative
CMD_STEP = np.array([0.1, 0.1, 0.2])
CMD_MAX  = np.array([0.8, 0.4, 1.0])


def projected_gravity(quat_wxyz):
    """world gravity [0,0,-1] expressed in the base frame (matches world_to_base)."""
    w, x, y, z = quat_wxyz
    return np.array([-2*(x*z + w*y), -2*(y*z - w*x), -(1 - 2*(x*x + y*y))])


def load_policy(lib_path, bin_path):
    lib = ctypes.CDLL(lib_path)
    lib.nn_init.restype = ctypes.c_int
    lib.nn_obs.restype = ctypes.c_int
    lib.nn_nu.restype = ctypes.c_int
    if lib.nn_init(bin_path.encode()) != 0:
        sys.exit(f"policy load failed: {bin_path}")
    obs_n, nu = lib.nn_obs(), lib.nn_nu()
    assert obs_n == 98 and nu == NU, f"policy shape mismatch obs={obs_n} nu={nu}"
    obs_buf = (ctypes.c_float * obs_n)()
    act_buf = (ctypes.c_float * nu)()

    def infer(obs):
        obs_buf[:] = obs.astype(np.float32)
        lib.nn_infer(obs_buf, act_buf)
        return np.frombuffer(act_buf, dtype=np.float32).copy()
    return infer


class G1Deploy:
    def __init__(self, infer, teleop):
        self.infer = infer
        self.teleop = teleop
        self.prev_action = np.zeros(NU)
        self.cmd = np.zeros(3)
        self.step = 0
        # SDK objects (imported lazily so the file at least parses without the SDK)
        from unitree_sdk2py.core.channel import ChannelPublisher, ChannelSubscriber
        from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowCmd_, LowState_
        from unitree_sdk2py.idl.default import unitree_hg_msg_dds__LowCmd_
        from unitree_sdk2py.utils.crc import CRC
        self.CRC = CRC()
        self.low_cmd = unitree_hg_msg_dds__LowCmd_()
        self.low_state = None
        self.mode_machine = 0
        self.pub = ChannelPublisher("rt/lowcmd", LowCmd_); self.pub.Init()
        self.sub = ChannelSubscriber("rt/lowstate", LowState_)
        self.sub.Init(self._on_state, 10)

    def _on_state(self, msg):
        self.low_state = msg
        self.mode_machine = msg.mode_machine

    def wait_for_state(self, timeout=5.0):
        t0 = time.time()
        while self.low_state is None:
            if time.time() - t0 > timeout:
                sys.exit("no LowState — check --net interface and that the robot is up")
            time.sleep(0.02)

    def _send(self, q, kp, kd, kd_only=False):
        self.low_cmd.mode_pr = 0          # PR (serial ankle) mode
        self.low_cmd.mode_machine = self.mode_machine
        for i in range(NU):
            m = self.low_cmd.motor_cmd[i]
            m.mode = 1                    # enable
            m.q   = 0.0 if kd_only else float(q[i])
            m.dq  = 0.0
            m.tau = 0.0
            m.kp  = 0.0 if kd_only else float(kp[i])
            m.kd  = float(kd[i])
        self.low_cmd.crc = self.CRC.Crc(self.low_cmd)
        self.pub.Write(self.low_cmd)

    def measured_q(self):
        return np.array([self.low_state.motor_state[i].q for i in range(NU)])

    def zero_torque(self, secs=1.0):
        print("[1/3] zero-torque (robot is limp — support it)…")
        t_end = time.time() + secs
        while time.time() < t_end:
            self._send(np.zeros(NU), np.zeros(NU), np.zeros(NU))
            time.sleep(CONTROL_DT)

    def move_to_home(self, secs=3.0):
        print(f"[2/3] moving to home pose over {secs:.0f}s…")
        q0 = self.measured_q()
        n = int(secs / CONTROL_DT)
        for k in range(n + 1):
            a = k / n
            q = (1 - a) * q0 + a * HOME
            self._send(q, KP, KD)
            time.sleep(CONTROL_DT)

    def build_obs(self):
        s = self.low_state
        ang = np.array([s.imu_state.gyroscope[i] for i in range(3)])
        quat = np.array([s.imu_state.quaternion[i] for i in range(4)])  # w,x,y,z
        q  = self.measured_q()
        dq = np.array([s.motor_state[i].dq for i in range(NU)])
        ph = 2 * math.pi * ((self.step % PHASE_PERIOD) / PHASE_PERIOD)
        obs = np.zeros(98, dtype=np.float64)
        obs[0:3]   = ANG_VEL_SCALE * ang
        obs[3:6]   = projected_gravity(quat)
        obs[6:9]   = self.cmd
        obs[9:38]  = q - HOME
        obs[38:67] = DOF_VEL_SCALE * dq
        obs[67:96] = self.prev_action
        obs[96], obs[97] = math.sin(ph), math.cos(ph)
        return obs

    def run(self):
        self.wait_for_state()
        self.zero_torque()
        self.move_to_home()
        kb = KeyTeleop() if self.teleop else None
        input("[3/3] home reached. ENTER to start the policy (Ctrl-C to stop)… ")
        print("policy running" + ("  — WASD to drive, space to stop" if kb else "  — walking in place"))
        try:
            next_t = time.time()
            while True:
                if kb:
                    self.cmd = kb.update(self.cmd)
                act = self.infer(self.build_obs())
                target = HOME.copy()
                for a in range(NU):
                    c = float(np.clip(act[a], -1.0, 1.0))
                    if a >= LEG_DOF:           # v3: legs only; waist+arms stay home
                        c = 0.0
                    self.prev_action[a] = c
                    t = HOME[a] + ACTION_SCALE * c
                    target[a] = np.clip(t, CTRL_RANGE[a, 0], CTRL_RANGE[a, 1])
                self._send(target, KP, KD)
                self.step += 1
                next_t += CONTROL_DT
                time.sleep(max(0.0, next_t - time.time()))
        except KeyboardInterrupt:
            print("\nstopping → damping")
            for _ in range(50):
                self._send(np.zeros(NU), np.zeros(NU), 2.0 * np.ones(NU), kd_only=True)
                time.sleep(CONTROL_DT)


class KeyTeleop:
    """Non-blocking WASD command teleop (raw stdin)."""
    def __init__(self):
        import termios, tty
        self.fd = sys.stdin.fileno()
        self.old = termios.tcgetattr(self.fd)
        tty.setcbreak(self.fd)

    def update(self, cmd):
        import select
        cmd = cmd.copy()
        while select.select([sys.stdin], [], [], 0)[0]:
            ch = sys.stdin.read(1).lower()
            if   ch == 'w': cmd[0] += CMD_STEP[0]
            elif ch == 's': cmd[0] -= CMD_STEP[0]
            elif ch == 'a': cmd[2] += CMD_STEP[2]
            elif ch == 'd': cmd[2] -= CMD_STEP[2]
            elif ch == ' ': cmd[:] = 0
        return np.clip(cmd, -CMD_MAX, CMD_MAX)

    def __del__(self):
        try:
            import termios
            termios.tcsetattr(self.fd, termios.TCSADRAIN, self.old)
        except Exception:
            pass


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    ap = argparse.ArgumentParser(description="Deploy nanoG1 on a real Unitree G1")
    ap.add_argument("--net", required=True, help="DDS network interface to the robot (e.g. eth0)")
    ap.add_argument("--bin", default=os.path.join(here, "..", "assets", "nanoG1.bin"))
    ap.add_argument("--lib", default=None, help="path to libnanog1policy.{so,dylib}")
    ap.add_argument("--teleop", action="store_true", help="WASD command teleop")
    args = ap.parse_args()

    lib = args.lib or os.path.join(here, "libnanog1policy." + ("dylib" if sys.platform == "darwin" else "so"))
    if not os.path.exists(lib):
        sys.exit(f"{lib} not found — run: bash deploy/build_policy.sh")

    from unitree_sdk2py.core.channel import ChannelFactoryInitialize
    ChannelFactoryInitialize(0, args.net)

    infer = load_policy(lib, os.path.abspath(args.bin))
    G1Deploy(infer, args.teleop).run()


if __name__ == "__main__":
    main()
