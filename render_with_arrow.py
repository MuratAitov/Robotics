"""
Render MuJoCo scene with direction arrow visualization.
Uses MuJoCo's built-in visualization to draw the direction vector.
"""
import os
import json
import numpy as np
import mujoco
import cv2

OUTPUT_DIR = "outputs/pose_with_arrow"
POSES_FILE = "FoundationPose/sneaker_poses.json"

# Sneaker position in world (from scene)
SNEAKER_POS = np.array([0, 0, 0.53])  # Where sneaker is placed in world_pose_test.xml

CAMERAS = ["front_cam", "side_right_cam", "side_left_cam", "top45_cam", "top_cam"]


def rotation_matrix_from_direction(direction):
    """Create rotation matrix to align Z-axis with direction vector."""
    direction = np.array(direction)
    direction = direction / np.linalg.norm(direction)

    # Default up vector
    up = np.array([0, 0, 1])

    # If direction is parallel to up, use different up
    if abs(np.dot(direction, up)) > 0.99:
        up = np.array([0, 1, 0])

    # Create orthonormal basis
    right = np.cross(up, direction)
    right = right / np.linalg.norm(right)

    new_up = np.cross(direction, right)
    new_up = new_up / np.linalg.norm(new_up)

    # Rotation matrix (direction becomes Z-axis)
    R = np.column_stack([right, new_up, direction])
    return R


def add_arrow_to_scene(model, data, start_pos, direction, length=0.15):
    """
    Add arrow visualization using MuJoCo sites.
    Returns the arrow endpoint for reference.
    """
    direction = np.array(direction)
    direction = direction / np.linalg.norm(direction)
    end_pos = start_pos + direction * length
    return start_pos, end_pos


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Load poses
    with open(POSES_FILE, 'r') as f:
        poses = json.load(f)

    # Create scene XML with arrow
    # We'll modify the scene to include arrow geometry

    for i, cam_name in enumerate(CAMERAS):
        if i >= len(poses):
            break

        direction = poses[i]['direction_vector']

        # Create modified XML with arrow
        arrow_length = 0.15
        dir_normalized = np.array(direction) / np.linalg.norm(direction)

        # Arrow start and end
        arrow_start = SNEAKER_POS + np.array([0, 0, 0.08])  # Slightly above sneaker
        arrow_end = arrow_start + dir_normalized * arrow_length
        arrow_mid = (arrow_start + arrow_end) / 2

        # Calculate arrow orientation (fromto format for capsule)
        xml_content = f'''<mujoco model="pose_visualization">
  <compiler angle="degree" coordinate="local"/>
  <option timestep="0.002" gravity="0 0 -9.81"/>

  <visual>
    <global offwidth="1280" offheight="720"/>
    <quality shadowsize="4096"/>
  </visual>

  <asset>
    <mesh name="sneakers_mesh" file="Sneakers/sneakers/sneakers.obj" scale="2.1 2.1 2.1"/>
    <texture name="sneakers_tex" type="2d" file="Sneakers/sneakers/cons_mat_baseColor.png"/>
    <material name="sneakers_mat" texture="sneakers_tex" specular="0.3"/>
    <texture name="table_tex" type="2d" builtin="flat" rgb1="0.6 0.55 0.5" width="64" height="64"/>
    <material name="table_mat" texture="table_tex" specular="0.2"/>
    <texture name="floor_tex" type="2d" builtin="checker" rgb1="0.3 0.3 0.32" rgb2="0.25 0.25 0.27" width="256" height="256"/>
    <material name="floor_mat" texture="floor_tex" texrepeat="8 8"/>
    <material name="arrow_mat" rgba="0 1 0 1" specular="0.8"/>
    <material name="arrow_tip_mat" rgba="1 0 0 1" specular="0.8"/>
  </asset>

  <worldbody>
    <geom name="floor" type="plane" size="2 2 0.1" material="floor_mat"/>

    <light pos="0 0 2.5" dir="0 0 -1" diffuse="0.9 0.9 0.9" specular="0.3 0.3 0.3" castshadow="true"/>
    <light pos="1.5 -1.5 2" dir="-0.5 0.5 -0.7" diffuse="0.5 0.5 0.5"/>
    <light pos="-1.5 -1.5 2" dir="0.5 0.5 -0.7" diffuse="0.5 0.5 0.5"/>

    <geom name="table_top" type="box" size="0.5 0.35 0.02" pos="0 0 0.5" material="table_mat"/>
    <geom name="table_leg1" type="cylinder" size="0.04 0.25" pos="0.42 0.28 0.25" material="table_mat"/>
    <geom name="table_leg2" type="cylinder" size="0.04 0.25" pos="-0.42 0.28 0.25" material="table_mat"/>
    <geom name="table_leg3" type="cylinder" size="0.04 0.25" pos="0.42 -0.28 0.25" material="table_mat"/>
    <geom name="table_leg4" type="cylinder" size="0.04 0.25" pos="-0.42 -0.28 0.25" material="table_mat"/>

    <body name="sneaker" pos="0 0 0.53">
      <geom name="sneaker_geom" type="mesh" mesh="sneakers_mesh" material="sneakers_mat" euler="-90 180 180"/>
    </body>

    <!-- DIRECTION ARROW -->
    <geom name="arrow_shaft" type="capsule"
          fromto="{arrow_start[0]} {arrow_start[1]} {arrow_start[2]} {arrow_end[0]} {arrow_end[1]} {arrow_end[2]}"
          size="0.008" material="arrow_mat"/>

    <!-- Arrow tip (sphere at end) -->
    <geom name="arrow_tip" type="sphere" pos="{arrow_end[0]} {arrow_end[1]} {arrow_end[2]}"
          size="0.02" material="arrow_tip_mat"/>

    <!-- Start point marker -->
    <geom name="arrow_start" type="sphere" pos="{arrow_start[0]} {arrow_start[1]} {arrow_start[2]}"
          size="0.015" rgba="1 1 0 1"/>

    <!-- Cameras -->
    <camera name="front_cam" pos="0 -0.9 0.7" xyaxes="1 0 0 0 0.3 1" fovy="50"/>
    <camera name="side_right_cam" pos="0.9 0 0.7" xyaxes="0 1 0 -0.3 0 1" fovy="50"/>
    <camera name="side_left_cam" pos="-0.9 0 0.7" xyaxes="0 -1 0 0.3 0 1" fovy="50"/>
    <camera name="top45_cam" pos="0 -0.7 1.1" xyaxes="1 0 0 0 0.7 0.7" fovy="50"/>
    <camera name="top_cam" pos="0 0 1.4" xyaxes="1 0 0 0 1 0" fovy="55"/>
  </worldbody>
</mujoco>'''

        # Save temp XML
        temp_xml = f"temp_scene_{cam_name}.xml"
        with open(temp_xml, 'w') as f:
            f.write(xml_content)

        # Load and render
        model = mujoco.MjModel.from_xml_path(temp_xml)
        data = mujoco.MjData(model)

        # Step simulation
        for _ in range(100):
            mujoco.mj_step(model, data)

        # Render
        renderer = mujoco.Renderer(model, height=720, width=1280)
        renderer.update_scene(data, camera=cam_name)
        rgb = renderer.render()

        # Add text overlay
        img = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        dir_text = f"Direction: ({direction[0]:.2f}, {direction[1]:.2f}, {direction[2]:.2f})"
        cv2.putText(img, dir_text, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(img, cam_name, (20, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

        # Save
        output_path = os.path.join(OUTPUT_DIR, f"{cam_name}_arrow.png")
        cv2.imwrite(output_path, img)
        print(f"Saved: {output_path}")

        # Clean up temp file
        os.remove(temp_xml)
        renderer.close()

    print(f"\nAll renders saved to {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
