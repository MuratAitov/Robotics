"""
Prepare sneaker mesh for FoundationPose.
Loads OBJ with texture and saves as GLB (includes embedded texture).
"""
import trimesh
import os
from PIL import Image
import numpy as np

SNEAKER_DIR = "Sneakers/sneakers"
OUTPUT_DIR = "FoundationPose/demo_data/sneaker/mesh"

def main():
    obj_path = os.path.join(SNEAKER_DIR, "sneakers.obj")
    texture_path = os.path.join(SNEAKER_DIR, "cons_mat_baseColor.jpg")
    output_path = os.path.join(OUTPUT_DIR, "sneakers_textured.glb")

    print(f"Loading mesh from {obj_path}")
    mesh = trimesh.load(obj_path)

    print(f"Mesh type: {type(mesh)}")
    print(f"Vertices: {len(mesh.vertices)}")
    print(f"Faces: {len(mesh.faces)}")

    # Check if mesh has texture
    if hasattr(mesh, 'visual'):
        print(f"Visual type: {type(mesh.visual)}")

        if isinstance(mesh.visual, trimesh.visual.texture.TextureVisuals):
            print("Mesh has texture visuals")
            if mesh.visual.material is not None:
                print(f"Material type: {type(mesh.visual.material)}")
                if hasattr(mesh.visual.material, 'image') and mesh.visual.material.image is not None:
                    print("Texture already embedded")
                else:
                    print("Loading texture manually...")
                    # Load texture image
                    if os.path.exists(texture_path):
                        img = Image.open(texture_path)
                        mesh.visual.material.image = img
                        print(f"Loaded texture: {img.size}")
                    else:
                        print(f"Texture not found: {texture_path}")
        else:
            print("Mesh has non-texture visuals, converting...")
            # Try to apply vertex colors instead
            if hasattr(mesh.visual, 'vertex_colors'):
                print(f"Has vertex colors: {mesh.visual.vertex_colors is not None}")

    # Export as GLB (includes embedded textures)
    print(f"Exporting to {output_path}")
    mesh.export(output_path)

    # Also export simple OBJ with vertex colors for fallback
    simple_mesh = mesh.copy()
    # Assign uniform gray color if no texture
    if not hasattr(simple_mesh.visual, 'vertex_colors') or simple_mesh.visual.vertex_colors is None:
        simple_mesh.visual.vertex_colors = np.tile([128, 128, 128, 255], (len(simple_mesh.vertices), 1))

    simple_path = os.path.join(OUTPUT_DIR, "sneakers_simple.obj")
    simple_mesh.export(simple_path)
    print(f"Exported simple mesh to {simple_path}")

    print("\nDone! Use one of these meshes with FoundationPose:")
    print(f"  - {output_path} (with texture)")
    print(f"  - {simple_path} (vertex colors)")

if __name__ == "__main__":
    main()
