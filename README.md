# 16-DOF Humanoid Hand
 # 16-DOF Humanoid Hand

 <video controls width="640" src="https://www.armanarghavani.com/humanoid_hand_4_fingers.mp4">Your browser does not support the video tag.</video>

 This repository contains design files, simulation assets, and demo code for a 16-degree-of-freedom (DOF) underactuated tendon-driven humanoid hand with forearm-mounted actuators. The goal of the project is to provide a mechanically simple hand that is easier to control than a fully-actuated design while achieving approximately 75% of human motion fidelity.

 **Highlights**
 - 16 total DOF across fingers and thumb
 - Underactuated tendon-driven finger actuation to reduce actuator count
 - Forearm-mounted actuators for compact hand wiring and reduced hand mass
 - Simulation-ready XML models and 3D part/mesh assets
 - Vision demo using MediaPipe hand landmarker (example: `hand_vision.py`)

 ## Mechanical Component

 ![Figure 1 — Mechanical parts and assembled hand](https://www.armanarghavani.com/hand.png)

 Figure 1: 3D-printed / CAD parts and assembled hand used in the project.

 ![Figure 2 — Tendon routing and connections diagram](https://www.armanarghavani.com/matlab_hand_sim.png)

 Figure 2: Tendon connections and routing diagram used for simulation and validation.

 Design notes and validation
 - MuJoCo Simulation: validated tendon routing
	 - Setup: Imported the hand/forearm CAD into MuJoCo (MJCF) and assembled the model with the forearm as the root, then cleaned up joint frames so each finger chain behaves correctly.
	 - Tendon routing: Used spatial tendons with sites + pulley wrap geoms (plus guide points) for realistic routing/moment arms, and added a fixed-tendon "winch" interface between spool rotation and the spatial tendons so the spools can reel cable in/out cleanly.
	 - Sensing: Logged tendon length and tension to verify reel-in/out + antagonistic behavior and quickly debug slack/preload and routing issues.

 ## Software Component

 - MuJoCo Simulation: scene and robot definitions are provided in `robot_hand_forearm_root.xml` and `scene_forearm_root.xml`. Use these files to run simulated experiments, validate tendon routing, and test control policies in a physics simulator.
 - MATLAB / Simulink: Simscape Multibody block diagram simulating joint ranges and limits, validating ~75% human motion capabilities.

 ## Quickstart
 1. Install Python dependencies (recommended in a virtualenv):

 ```bash
 python3 -m venv .venv
 source .venv/bin/activate
 pip3 install --upgrade pip
 pip3 install opencv-python mediapipe
 ```

 2. Place the MediaPipe task model file next to the demo script:

 - Ensure `hand_landmarker.task` exists at repository root (next to `hand_vision.py`). If you don't have the file, obtain the correct MediaPipe hand landmarker task file or generate it following MediaPipe instructions.

 3. Run the vision demo (opens webcam):

 ```bash
 python3 hand_vision.py
 ```

 ## Troubleshooting
 - The `hand_vision.py` demo uses the MediaPipe Tasks API and OpenCV. If you see an error about loading `hand_landmarker.task` (e.g., "Unable to get file size"), confirm the file exists and is not corrupted. Place it in the same folder as `hand_vision.py`.
 - On macOS you may need to allow camera access for your terminal or Python app in System Preferences > Security & Privacy.
 - If `pip3` is not available, use your system package manager or `python3 -m pip install ...`.

 ## File structure
 - `robot_hand_forearm_root.xml`, `scene_forearm_root.xml` — MuJoCo scene and robot files for simulation
 - `assets/` — part definitions used by the hand model
 - `meshes/` — 3D geometry used by the parts (if present)
 - `hand_vision.py`, `hand_landmarker.task` — example vision demo and MediaPipe model file (model file should be placed next to the script)
 - `hand_test.py` — tests / quick checks (if present)

 ## How To Contribute
 - Create issues or PRs with improvements to the mechanical model, control strategy, or demos.
