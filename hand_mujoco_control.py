"""
Hand Vision Tracking + MuJoCo Control Integration
Maps real-time hand landmarks from webcam to robotic hand joint angles in MuJoCo simulation
"""

import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import mujoco
import mujoco.viewer
import math

# MediaPipe setup
MODEL_PATH = "hand_landmarker.task"

base_options = python.BaseOptions(model_asset_path=MODEL_PATH)
options = vision.HandLandmarkerOptions(
    base_options=base_options,
    num_hands=1,
    min_hand_detection_confidence=0.5,
    min_tracking_confidence=0.5,
)
detector = vision.HandLandmarker.create_from_options(options)

# MuJoCo setup
model = mujoco.MjModel.from_xml_path("scene_forearm_root.xml")
data = mujoco.MjData(model)

# Joint name mappings (16 DOF total)
JOINT_NAMES = {
    # Index finger (3 joints)
    'index_mcp': 0,
    'index_pip': 1,
    'index_dip': 2,
    # Middle finger (3 joints)
    'middle_mcp': 3,
    'middle_pip': 4,
    'middle_dip': 5,
    # Ring finger (3 joints)
    'ring_mcp': 6,
    'ring_pip': 7,
    'ring_dip': 8,
    # Pinky finger (3 joints)
    'pinky_mcp': 9,
    'pinky_pip': 10,
    'pinky_dip': 11,
    # Thumb (4 joints)
    'thumb_opp': 12,
    'thumb_mcp': 13,
    'thumb_pip': 14,
    'thumb_dip': 15,
}

# MediaPipe hand landmark indices
# https://developers.google.com/mediapipe/solutions/vision/hand_landmarker
WRIST = 0
THUMB_CMC = 1
THUMB_MCP = 2
THUMB_IP = 3
THUMB_TIP = 4
INDEX_MCP = 5
INDEX_PIP = 6
INDEX_DIP = 7
INDEX_TIP = 8
MIDDLE_MCP = 9
MIDDLE_PIP = 10
MIDDLE_DIP = 11
MIDDLE_TIP = 12
RING_MCP = 13
RING_PIP = 14
RING_DIP = 15
RING_TIP = 16
PINKY_MCP = 17
PINKY_PIP = 18
PINKY_DIP = 19
PINKY_TIP = 20


def calculate_angle_3d(p1, p2, p3):
    """Calculate angle at p2 formed by points p1-p2-p3 in 3D space"""
    v1 = np.array([p1.x - p2.x, p1.y - p2.y, p1.z - p2.z])
    v2 = np.array([p3.x - p2.x, p3.y - p2.y, p3.z - p2.z])
    
    # Normalize vectors
    v1_norm = np.linalg.norm(v1)
    v2_norm = np.linalg.norm(v2)
    
    if v1_norm < 1e-6 or v2_norm < 1e-6:
        return 0.0
    
    v1 = v1 / v1_norm
    v2 = v2 / v2_norm
    
    # Calculate angle
    dot_product = np.clip(np.dot(v1, v2), -1.0, 1.0)
    angle = np.arccos(dot_product)
    
    return angle


def landmarks_to_joint_angles(landmarks):
    """
    Convert MediaPipe hand landmarks to robot joint angles
    Returns a dict of joint_name: angle_in_radians
    """
    if not landmarks or len(landmarks) < 21:
        return None
    
    joint_angles = {}
    
    # For each finger, calculate flexion angles
    # Note: MediaPipe gives us extension (180° = straight), 
    # MuJoCo wants flexion (0 = straight, negative = bent for some joints)
    
    # Index finger
    wrist = landmarks[WRIST]
    index_mcp_lm = landmarks[INDEX_MCP]
    index_pip_lm = landmarks[INDEX_PIP]
    index_dip_lm = landmarks[INDEX_DIP]
    index_tip = landmarks[INDEX_TIP]
    
    # MCP joint (metacarpophalangeal) - measure from wrist to mcp to pip
    mcp_angle = calculate_angle_3d(wrist, index_mcp_lm, index_pip_lm)
    joint_angles['index_mcp'] = -(np.pi - mcp_angle)  # Negative because range is [-1.57, 0]
    
    # PIP joint (proximal interphalangeal)
    pip_angle = calculate_angle_3d(index_mcp_lm, index_pip_lm, index_dip_lm)
    joint_angles['index_pip'] = np.pi - pip_angle  # Positive flexion, range [0, 1.92]
    
    # DIP joint (distal interphalangeal)
    dip_angle = calculate_angle_3d(index_pip_lm, index_dip_lm, index_tip)
    joint_angles['index_dip'] = np.pi - dip_angle  # Positive flexion, range [0, 1.22]
    
    # Middle finger
    middle_mcp_lm = landmarks[MIDDLE_MCP]
    middle_pip_lm = landmarks[MIDDLE_PIP]
    middle_dip_lm = landmarks[MIDDLE_DIP]
    middle_tip = landmarks[MIDDLE_TIP]
    
    mcp_angle = calculate_angle_3d(wrist, middle_mcp_lm, middle_pip_lm)
    joint_angles['middle_mcp'] = -(np.pi - mcp_angle)
    
    pip_angle = calculate_angle_3d(middle_mcp_lm, middle_pip_lm, middle_dip_lm)
    joint_angles['middle_pip'] = np.pi - pip_angle
    
    dip_angle = calculate_angle_3d(middle_pip_lm, middle_dip_lm, middle_tip)
    joint_angles['middle_dip'] = np.pi - dip_angle
    
    # Ring finger
    ring_mcp_lm = landmarks[RING_MCP]
    ring_pip_lm = landmarks[RING_PIP]
    ring_dip_lm = landmarks[RING_DIP]
    ring_tip = landmarks[RING_TIP]
    
    mcp_angle = calculate_angle_3d(wrist, ring_mcp_lm, ring_pip_lm)
    joint_angles['ring_mcp'] = -(np.pi - mcp_angle)
    
    pip_angle = calculate_angle_3d(ring_mcp_lm, ring_pip_lm, ring_dip_lm)
    joint_angles['ring_pip'] = np.pi - pip_angle
    
    dip_angle = calculate_angle_3d(ring_pip_lm, ring_dip_lm, ring_tip)
    joint_angles['ring_dip'] = np.pi - dip_angle
    
    # Pinky finger
    pinky_mcp_lm = landmarks[PINKY_MCP]
    pinky_pip_lm = landmarks[PINKY_PIP]
    pinky_dip_lm = landmarks[PINKY_DIP]
    pinky_tip = landmarks[PINKY_TIP]
    
    mcp_angle = calculate_angle_3d(wrist, pinky_mcp_lm, pinky_pip_lm)
    joint_angles['pinky_mcp'] = -(np.pi - mcp_angle)
    
    pip_angle = calculate_angle_3d(pinky_mcp_lm, pinky_pip_lm, pinky_dip_lm)
    joint_angles['pinky_pip'] = np.pi - pip_angle
    
    dip_angle = calculate_angle_3d(pinky_pip_lm, pinky_dip_lm, pinky_tip)
    joint_angles['pinky_dip'] = -(np.pi - dip_angle)  # Pinky DIP is negative range
    
    # Thumb (more complex - opposition + flexion)
    thumb_cmc = landmarks[THUMB_CMC]
    thumb_mcp_lm = landmarks[THUMB_MCP]
    thumb_ip = landmarks[THUMB_IP]
    thumb_tip = landmarks[THUMB_TIP]
    
    # Opposition angle (rotation away from palm)
    opp_angle = calculate_angle_3d(wrist, thumb_cmc, index_mcp_lm)
    joint_angles['thumb_opp'] = (opp_angle - np.pi/2) * 0.5  # Scale to reasonable range
    
    # Thumb MCP
    mcp_angle = calculate_angle_3d(thumb_cmc, thumb_mcp_lm, thumb_ip)
    joint_angles['thumb_mcp'] = (np.pi - mcp_angle) - 0.5  # Center around 0
    
    # Thumb PIP (IP joint)
    pip_angle = calculate_angle_3d(thumb_mcp_lm, thumb_ip, thumb_tip)
    joint_angles['thumb_pip'] = -(np.pi - pip_angle)  # Negative range
    
    # Thumb DIP (tip joint) - estimate from overall curl
    joint_angles['thumb_dip'] = joint_angles['thumb_pip'] * 0.6  # Coupled motion
    
    return joint_angles


def apply_joint_angles_to_mujoco(data, joint_angles):
    """Apply computed joint angles to MuJoCo model"""
    if joint_angles is None:
        return
    
    for joint_name, angle in joint_angles.items():
        if joint_name in JOINT_NAMES:
            joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
            if joint_id >= 0:
                # Get joint limits
                jnt_range = model.jnt_range[joint_id]
                # Clamp angle to joint limits
                clamped_angle = np.clip(angle, jnt_range[0], jnt_range[1])
                # Set the joint position
                qpos_idx = model.jnt_qposadr[joint_id]
                data.qpos[qpos_idx] = clamped_angle


# Camera capture setup
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

print("Starting Hand Tracking + MuJoCo Control")
print("Show your hand to the camera to control the robotic hand")
print("Press ESC in camera window to quit")

# Launch MuJoCo viewer
viewer = mujoco.viewer.launch_passive(model, data)

try:
    while viewer.is_running():
        # Capture frame from camera
        ok, frame = cap.read()
        if not ok:
            break
        
        # Convert to RGB for MediaPipe
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        
        # Detect hand landmarks
        result = detector.detect(mp_image)
        
        # Process landmarks and control robot
        if result.hand_landmarks:
            hand = result.hand_landmarks[0]  # First hand only
            
            # Draw landmarks on camera view
            for lm in hand:
                x = int(lm.x * frame.shape[1])
                y = int(lm.y * frame.shape[0])
                cv2.circle(frame, (x, y), 3, (0, 255, 0), -1)
            
            # Calculate joint angles from landmarks
            joint_angles = landmarks_to_joint_angles(hand)
            
            # Apply to MuJoCo simulation
            apply_joint_angles_to_mujoco(data, joint_angles)
        
        # Display camera feed
        cv2.imshow("Hand Tracking", frame)
        
        # Step MuJoCo simulation
        mujoco.mj_step(model, data)
        
        # Sync viewer
        viewer.sync()
        
        # Check for exit
        if (cv2.waitKey(1) & 0xFF) == 27:  # ESC
            break
            
finally:
    viewer.close()

cap.release()
cv2.destroyAllWindows()
print("Stopped hand tracking control")
