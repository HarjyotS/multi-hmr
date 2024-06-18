import bpy
import os

# Set the path to the directory containing the .glb files
mesh_directory = r"D:\multi-hmr\demo_out\meshes"
output_blend_path = r"D:\multi-hmr\demo_out\meshesanimation.blend"
frame_duration = 1  # Set the duration of each frame in the animation

# Clear all objects in the scene
bpy.ops.wm.read_factory_settings(use_empty=True)

# Set the scene properties
scene = bpy.context.scene
scene.render.fps = 24
scene.frame_start = 1

# Ensure the main collection exists
if "Collection" not in bpy.data.collections:
    main_collection = bpy.data.collections.new("Collection")
    bpy.context.scene.collection.children.link(main_collection)
else:
    main_collection = bpy.data.collections["Collection"]

# List all .glb files in the directory
mesh_files = sorted([f for f in os.listdir(mesh_directory) if f.endswith(".glb")])

# Set the end frame based on the number of mesh files
scene.frame_end = len(mesh_files) * frame_duration

# Import each .glb file and set it to the corresponding frame
for i, mesh_file in enumerate(mesh_files):
    frame_number = i * frame_duration + 1
    bpy.context.scene.frame_set(frame_number)

    # Import the .glb file
    filepath = os.path.join(mesh_directory, mesh_file)
    bpy.ops.import_scene.gltf(filepath=filepath)

    # Move imported objects to a collection named after the frame number
    imported_objects = bpy.context.selected_objects
    collection_name = f"Frame_{frame_number}"
    if collection_name not in bpy.data.collections:
        collection = bpy.data.collections.new(collection_name)
        bpy.context.scene.collection.children.link(collection)
    else:
        collection = bpy.data.collections[collection_name]

    for obj in imported_objects:
        if obj.name in main_collection.objects:
            main_collection.objects.unlink(obj)
        collection.objects.link(obj)

    # Set the visibility of the imported objects to the corresponding frame range
    for obj in imported_objects:
        obj.hide_render = True
        obj.hide_viewport = True
        obj.keyframe_insert(data_path="hide_render", frame=frame_number - 1)
        obj.keyframe_insert(data_path="hide_viewport", frame=frame_number - 1)
        obj.hide_render = False
        obj.hide_viewport = False
        obj.keyframe_insert(data_path="hide_render", frame=frame_number)
        obj.keyframe_insert(data_path="hide_viewport", frame=frame_number)
        obj.hide_render = True
        obj.hide_viewport = True
        obj.keyframe_insert(
            data_path="hide_render", frame=frame_number + frame_duration
        )
        obj.keyframe_insert(
            data_path="hide_viewport", frame=frame_number + frame_duration
        )

# Save the Blender project as a .blend file
bpy.ops.wm.save_as_mainfile(filepath=output_blend_path)
