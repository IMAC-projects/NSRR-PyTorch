import bpy 

bpy.ops.mesh.primitive_cube_add(size=2, enter_editmode=False, location=(0, 0, 0))
scene = bpy.context.scene
obj = bpy.data.objects["Cube"]
obj.location = (0,0,0)




### animation
positions = (0,0,2),(0,1,2),(3,2,1),(3,4,1),(1,2,1)

# start with frame 0
number_of_frame = 0  
for pos in positions:

    # now we will describe frame with number $number_of_frame
    scene.frame_set(number_of_frame)

    # set new location and rotation for object 
    obj.location = pos
    obj.keyframe_insert(data_path="location", index=-1)
    obj.keyframe_insert(data_path="rotation_euler", index=-1)
    obj.rotation_euler = pos
    # move next 10 frames forward - Blender will figure out what to do between this time
    number_of_frame += 10