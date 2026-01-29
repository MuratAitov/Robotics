"""
Generate RGB + Depth + Mask data from MuJoCo for FoundationPose.
Outputs data in FoundationPose demo format.
"""
import os
import sys
import numpy as np
import mujoco
import cv2

# Output directory for FoundationPose
OUTPUT_DIR = "FoundationPose/demo_data/sneaker"

def main():
    # Use pose test scene by default, or specify as argument
    scene_file = sys.argv[1] if len(sys.argv) > 1 else "world_pose_test.xml"
    print(f"Loading scene: {scene_file}")

    model = mujoco.MjModel.from_xml_path(scene_file)
    data = mujoco.MjData(model)

    # Simulate to let objects settle
    for _ in range(500):
        mujoco.mj_step(model, data)

    # Create renderer with depth enabled
    height, width = 720, 1280
    renderer = mujoco.Renderer(model, height=height, width=width)

    # Create output directories
    rgb_dir = os.path.join(OUTPUT_DIR, "rgb")
    depth_dir = os.path.join(OUTPUT_DIR, "depth")
    mask_dir = os.path.join(OUTPUT_DIR, "masks")
    os.makedirs(rgb_dir, exist_ok=True)
    os.makedirs(depth_dir, exist_ok=True)
    os.makedirs(mask_dir, exist_ok=True)

    # Get all camera names from scene
    cameras = []
    for i in range(model.ncam):
        cam_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_CAMERA, i)
        if cam_name:
            cameras.append(cam_name)

    print(f"Found {len(cameras)} cameras: {cameras}")

    for i, cam_name in enumerate(cameras):
        print(f"Rendering {cam_name}...")

        # Render RGB
        renderer.update_scene(data, camera=cam_name)
        rgb = renderer.render()

        # Save RGB (convert to BGR for OpenCV)
        rgb_path = os.path.join(rgb_dir, f"{i:06d}.png")
        cv2.imwrite(rgb_path, cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))

        # Render Depth
        renderer.enable_depth_rendering()
        renderer.update_scene(data, camera=cam_name)
        depth = renderer.render()
        renderer.disable_depth_rendering()

        # Normalize depth for visualization and save
        # FoundationPose expects depth in millimeters as uint16
        depth_mm = (depth * 1000).astype(np.uint16)
        depth_path = os.path.join(depth_dir, f"{i:06d}.png")
        cv2.imwrite(depth_path, depth_mm)

        # Render segmentation mask
        renderer.enable_segmentation_rendering()
        renderer.update_scene(data, camera=cam_name)
        seg = renderer.render()
        renderer.disable_segmentation_rendering()

        # Extract object IDs (first channel contains geom IDs)
        # Create binary mask for sneaker objects
        geom_ids = seg[:, :, 0]

        # Find sneaker geom IDs (objects with "sneaker" in name)
        sneaker_mask = np.zeros((height, width), dtype=np.uint8)
        for geom_id in range(model.ngeom):
            geom_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, geom_id)
            if geom_name and "sneaker" in geom_name.lower():
                sneaker_mask[geom_ids == geom_id] = 255

        # If no specific sneaker geoms found, use all non-background
        if sneaker_mask.max() == 0:
            # Exclude background (usually ID 0 or -1)
            sneaker_mask = ((geom_ids > 0) * 255).astype(np.uint8)

        mask_path = os.path.join(mask_dir, f"{i:06d}.png")
        cv2.imwrite(mask_path, sneaker_mask)

        print(f"  Saved: {rgb_path}, {depth_path}, {mask_path}")

    # Save camera intrinsics (approximate for MuJoCo default camera)
    # FoundationPose needs K matrix
    fovy = model.vis.global_.fovy  # Field of view in degrees
    fovy_rad = np.deg2rad(fovy)

    # Compute focal length from FOV
    fy = height / (2 * np.tan(fovy_rad / 2))
    fx = fy  # Assume square pixels
    cx = width / 2
    cy = height / 2

    K = np.array([
        [fx, 0, cx],
        [0, fy, cy],
        [0, 0, 1]
    ])

    intrinsics_path = os.path.join(OUTPUT_DIR, "cam_K.txt")
    np.savetxt(intrinsics_path, K)
    print(f"\nCamera intrinsics saved to {intrinsics_path}")
    print(f"K matrix:\n{K}")

    print(f"\nGenerated {len(cameras)} frames for FoundationPose")
    print(f"Data saved to: {OUTPUT_DIR}")

if __name__ == "__main__":
    main()
