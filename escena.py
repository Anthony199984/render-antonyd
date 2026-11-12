"""Escena de prueba por código: piso, esfera, cubo y una luz. Uso:
blender -b --factory-startup -P escena.py -- <CYCLES|EEVEE> salida.png"""
import bpy, sys, time, math
motor, salida = sys.argv[sys.argv.index('--') + 1:][:2]
bpy.ops.wm.read_factory_settings(use_empty=True)
esc = bpy.context.scene
def mat(nombre, color, rugoso):
    m = bpy.data.materials.new(nombre); m.use_nodes = True
    b = m.node_tree.nodes['Principled BSDF']; b.inputs['Base Color'].default_value = (*color, 1); b.inputs['Roughness'].default_value = rugoso
    return m
bpy.ops.mesh.primitive_plane_add(size=20); bpy.context.object.data.materials.append(mat('piso', (.55, .55, .58), .6))
bpy.ops.mesh.primitive_uv_sphere_add(radius=1, location=(-1.1, 0, 1), segments=64, ring_count=32); bpy.ops.object.shade_smooth()
bpy.context.object.data.materials.append(mat('esfera', (.8, .15, .1), .25))
bpy.ops.mesh.primitive_cube_add(size=1.4, location=(1.2, .6, .7), rotation=(0, 0, math.radians(30))); bpy.context.object.data.materials.append(mat('cubo', (.1, .35, .8), .4))
bpy.ops.object.light_add(type='AREA', location=(3, -3, 5)); l = bpy.context.object; l.data.energy = 1500; l.data.size = 3
l.rotation_euler = (math.radians(40), 0, math.radians(45))
w = bpy.data.worlds.new('mundo'); w.use_nodes = True; w.node_tree.nodes['Background'].inputs['Color'].default_value = (.05, .06, .08, 1); esc.world = w
bpy.ops.object.camera_add(location=(0, -7.5, 3.2), rotation=(math.radians(73), 0, 0)); esc.camera = bpy.context.object
esc.render.resolution_x, esc.render.resolution_y, esc.render.resolution_percentage = 1080, 1920, 100
esc.render.filepath = salida; esc.render.image_settings.file_format = 'PNG'
motores = [i.identifier for i in esc.render.bl_rna.properties['engine'].enum_items]
if motor == 'CYCLES':
    esc.render.engine = 'CYCLES'; esc.cycles.device = 'CPU'; esc.cycles.samples = 128
    esc.cycles.use_denoising = True; esc.cycles.denoiser = 'OPENIMAGEDENOISE'
else:
    esc.render.engine = 'BLENDER_EEVEE_NEXT' if 'BLENDER_EEVEE_NEXT' in motores else 'BLENDER_EEVEE'
print('MOTOR', esc.render.engine, 'versión', bpy.app.version_string, flush=True)
a = time.time(); bpy.ops.render.render(write_still=True)
print(f'TIEMPO_CUADRO {time.time() - a:.1f}', flush=True)
