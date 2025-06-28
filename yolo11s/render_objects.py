import blenderproc as bproc
import os
import random

bproc.init()

# Load all .obj models
model_dir = "models"
objects = []
for object_id, filename in enumerate(os.listdir(model_dir)):
    if filename.endswith(".obj"):
        obj = bproc.loader.load_obj(os.path.join(model_dir, filename))[0]
        obj.set_cp("category_id", object_id)
        obj.set_location([0, 0, 0])
        obj.set_rotation_euler([0, 0, 0])
        obj.set_scale([1, 1, 1])
        objects.append(obj)

# Add lighting
sun_light = bproc.types.Light()
sun_light.set_type("SUN")
sun_light.set_location([0, 0, 3])
sun_light.set_energy(2.0)

ambient_light = bproc.types.Light()
ambient_light.set_type("AREA")
ambient_light.set_location([0, 0, 3])
ambient_light.set_energy(0.5)

# Camera intrinsics
bproc.camera.set_resolution(640, 480)
bproc.camera.set_intrinsics_from_blender_params(lens=35)

# Output
output_dir = "bproc_output/train_pbr"
num_frames = 100

# Enable rendering outputs
bproc.renderer.enable_depth_output(activate_antialiasing=False)
bproc.renderer.enable_segmentation_output(map_by="category_id")

for i in range(num_frames):
    # Generate random camera pose
    location = bproc.sampler.shell(
        center=[0, 0, 0],
        radius_min=0.4, radius_max=0.7,
        elevation_min=5, elevation_max=85
    )
    cam2world = bproc.math.build_transformation_mat(location, [0, 0, 0])
    bproc.camera.add_camera_pose(cam2world)

    # Random object rotation (keep centered)
    for obj in objects:
        obj.set_location([0, 0, 0])
        obj.set_rotation_euler([0, 0, random.uniform(0, 2 * 3.14)])
        obj.set_scale([1, 1, 1])

    # Render and write per-frame output
    data = bproc.renderer.render()
    bproc.writer.write_bop(
        output_dir,
        target_objects=objects,
        depths=data["depth"],
        colors=data["colors"],
        append_to_existing_output=True
    )

    # Clear camera pose for next frame
    bproc.camera._cam_poses = []

print(f"✅ Done! Output saved to: {output_dir}")
