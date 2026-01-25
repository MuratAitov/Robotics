"""
Preview and test OBJ models in MuJoCo.
Run: python preview_models.py

Controls in viewer:
- Mouse drag: rotate view
- Scroll: zoom
- Double-click: reset view
- ESC: close
"""

import mujoco
import mujoco.viewer
import os
import tempfile
import numpy as np

# Models configuration: path, scale, euler rotation (degrees), position
MODELS = {
    "brown_sneakers": {
        "path": "Sneakers/brown_sneakers/brown_sneakers.obj",
        "texture": "Sneakers/brown_sneakers/rb_r_diffuse.png",
        "scale": 1.0,
        "euler": [0, 0, 0],  # rotation X, Y, Z in degrees
        "pos": [0, 0, 0.1],
    },
    "converse": {
        "path": "Sneakers/converse__free/converse__free.obj",
        "texture": "Sneakers/converse__free/material0_baseColor.jpg",
        "scale": 1.0,
        "euler": [0, 0, 0],
        "pos": [0, 0, 0.1],
    },
    "sneaker": {
        "path": "Sneakers/sneaker/sneaker.obj",
        "texture": "Sneakers/sneaker/RS_Material_8_baseColor.png",
        "scale": 1.0,
        "euler": [0, 0, 0],
        "pos": [0, 0, 0.1],
    },
    "sneakers": {
        "path": "Sneakers/sneakers/sneakers.obj",
        "texture": "Sneakers/sneakers/cons_mat_baseColor.jpg",
        "scale": 1.0,
        "euler": [0, 0, 0],
        "pos": [0, 0, 0.1],
    },
}


def euler_to_quat(roll, pitch, yaw):
    """Convert euler angles (degrees) to quaternion [w, x, y, z]."""
    r = np.radians(roll)
    p = np.radians(pitch)
    y = np.radians(yaw)

    cr, sr = np.cos(r/2), np.sin(r/2)
    cp, sp = np.cos(p/2), np.sin(p/2)
    cy, sy = np.cos(y/2), np.sin(y/2)

    w = cr * cp * cy + sr * sp * sy
    x = sr * cp * cy - cr * sp * sy
    y = cr * sp * cy + sr * cp * sy
    z = cr * cp * sy - sr * sp * cy

    return [w, x, y, z]


def create_xml(model_name, config):
    """Generate MuJoCo XML for a single model."""

    euler = config["euler"]
    quat = euler_to_quat(euler[0], euler[1], euler[2])
    quat_str = f"{quat[0]:.6f} {quat[1]:.6f} {quat[2]:.6f} {quat[3]:.6f}"
    pos = config["pos"]
    pos_str = f"{pos[0]} {pos[1]} {pos[2]}"
    scale = config["scale"]

    xml = f'''<mujoco model="{model_name}">
  <compiler angle="radian" coordinate="local"/>
  <option timestep="0.002" gravity="0 0 -9.81"/>

  <asset>
    <texture name="grid" type="2d" builtin="checker" width="512" height="512"
             rgb1="0.8 0.8 0.8" rgb2="0.6 0.6 0.6"/>
    <material name="floor_mat" texture="grid" texrepeat="10 10" reflectance="0.1"/>

    <texture name="obj_tex" type="2d" file="{config["texture"]}"/>
    <material name="obj_mat" texture="obj_tex"/>

    <mesh name="obj_mesh" file="{config["path"]}" scale="{scale} {scale} {scale}"/>
  </asset>

  <worldbody>
    <light pos="0 0 3" dir="0 0 -1" diffuse="1 1 1" specular="0.3 0.3 0.3"/>
    <light pos="2 2 2" dir="-1 -1 -1" diffuse="0.5 0.5 0.5"/>

    <geom type="plane" size="2 2 0.1" material="floor_mat"/>

    <body name="object" pos="{pos_str}" quat="{quat_str}">
      <freejoint/>
      <geom type="mesh" mesh="obj_mesh" material="obj_mat"/>
    </body>

    <camera name="front" pos="0.5 0 0.3" xyaxes="0 1 0 0 0 1" fovy="60"/>
    <camera name="side" pos="0 0.5 0.3" xyaxes="-1 0 0 0 0 1" fovy="60"/>
    <camera name="top" pos="0 0 0.8" xyaxes="1 0 0 0 1 0" fovy="60"/>
  </worldbody>
</mujoco>'''
    return xml


def preview_model(model_name, config):
    """Open viewer for a single model."""
    print(f"\n{'='*50}")
    print(f"Loading: {model_name}")
    print(f"  Path: {config['path']}")
    print(f"  Scale: {config['scale']}")
    print(f"  Rotation (euler deg): {config['euler']}")
    print(f"  Position: {config['pos']}")
    print(f"{'='*50}")

    xml = create_xml(model_name, config)

    try:
        model = mujoco.MjModel.from_xml_string(xml)
        data = mujoco.MjData(model)

        # Step simulation a bit to let object settle
        for _ in range(100):
            mujoco.mj_step(model, data)

        print("Opening viewer... (close window to continue)")
        mujoco.viewer.launch(model, data)

    except Exception as e:
        print(f"ERROR loading {model_name}: {e}")
        return False

    return True


def save_preview_image(model_name, config, output_path):
    """Render and save preview image."""
    xml = create_xml(model_name, config)

    try:
        model = mujoco.MjModel.from_xml_string(xml)
        data = mujoco.MjData(model)

        # Let object settle
        for _ in range(200):
            mujoco.mj_step(model, data)

        renderer = mujoco.Renderer(model, height=480, width=640)
        renderer.update_scene(data, camera="front")
        rgb = renderer.render()

        import cv2
        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        cv2.imwrite(output_path, bgr)
        print(f"Saved: {output_path}")
        return True

    except Exception as e:
        print(f"ERROR: {e}")
        return False


def main():
    # Change to script directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir)

    print("MuJoCo Model Previewer")
    print("=" * 50)
    print("\nAvailable models:")
    for i, name in enumerate(MODELS.keys(), 1):
        print(f"  {i}. {name}")
    print(f"  {len(MODELS)+1}. Preview ALL (save images)")
    print(f"  0. Exit")

    while True:
        try:
            choice = input("\nSelect model (number): ").strip()
            if choice == "0":
                break

            choice = int(choice)

            if choice == len(MODELS) + 1:
                # Save all previews
                os.makedirs("previews", exist_ok=True)
                for name, config in MODELS.items():
                    save_preview_image(name, config, f"previews/{name}.png")
                print("\nAll previews saved to 'previews/' folder")
            elif 1 <= choice <= len(MODELS):
                model_name = list(MODELS.keys())[choice - 1]
                preview_model(model_name, MODELS[model_name])
            else:
                print("Invalid choice")

        except ValueError:
            print("Enter a number")
        except KeyboardInterrupt:
            break

    print("\nDone!")


if __name__ == "__main__":
    main()
