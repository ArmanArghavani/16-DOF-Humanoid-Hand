# curl_extend_demo.py
# pip install mujoco
# Run: python curl_extend_demo.py

import time
import math
import mujoco
import mujoco.viewer

XML_PATH = "scene_forearm_root.xml"  # or your scene file, e.g. "scene_forearm_root.xml"

# Actuator names you told me to control
FLEX = ["motor_i1", "motor_m1", "motor_r1", "motor_p1"]
EXT  = ["motor_i2", "motor_m2", "motor_r2", "motor_p2"]

# If curling happens when motor goes negative in your model, set this to -1
CURL_SIGN = -1.0

# Motion timing
FREQ_HZ = 0.25  # 4 seconds per full curl/extend cycle

def get_actuator_id(model, name: str) -> int:
    act_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)
    if act_id < 0:
        raise ValueError(f"Actuator '{name}' not found. Check spelling vs your <actuator> names.")
    return act_id

def mid_and_amp(ctrl_min, ctrl_max, frac=0.85):
    """Drive within ctrlrange safely: mid +/- frac*(range/2)."""
    mid = 0.5 * (ctrl_min + ctrl_max)
    half = 0.5 * (ctrl_max - ctrl_min)
    amp = frac * half
    return mid, amp

def main():
    model = mujoco.MjModel.from_xml_path(XML_PATH)
    data = mujoco.MjData(model)

    # Map names -> ids
    flex_ids = [get_actuator_id(model, n) for n in FLEX]
    ext_ids  = [get_actuator_id(model, n) for n in EXT]

    # Precompute per-actuator mid/amp from ctrlrange
    mids = {}
    amps = {}
    for n in FLEX + EXT:
        i = get_actuator_id(model, n)
        if model.actuator_ctrllimited[i]:
            cmin, cmax = model.actuator_ctrlrange[i]
        else:
            # If not ctrl-limited, pick something reasonable (you can change this)
            cmin, cmax = (-1.0, 1.0)
        mid, amp = mid_and_amp(cmin, cmax, frac=0.85)
        mids[i] = mid
        amps[i] = amp
        print(f"{n:10s} id={i:2d} ctrlrange=({cmin:.3f},{cmax:.3f})  driving: mid={mid:.3f} amp={amp:.3f}")

    mujoco.mj_forward(model, data)

    with mujoco.viewer.launch_passive(model, data) as viewer:
        t0 = time.time()
        while viewer.is_running():
            t = time.time() - t0
            s = math.sin(2.0 * math.pi * FREQ_HZ * t)

            # Curl/extend command:
            # flex = mid + sign*amp*sine
            # ext  = mid - sign*amp*sine   (opposes flex)
            for fid in flex_ids:
                data.ctrl[fid] = mids[fid] + CURL_SIGN * amps[fid] * s
            for eid in ext_ids:
                data.ctrl[eid] = mids[eid] - CURL_SIGN * amps[eid] * s

            mujoco.mj_step(model, data)
            viewer.sync()

if __name__ == "__main__":
    main()
