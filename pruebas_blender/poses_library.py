#para ponerlo en script de blender y ejecutarlo
# guarda la pose actual del armature activo en una biblioteca JSON
'''
PARA EJECUTAR DESDE POWERSHELL:

Start-Process -FilePath "C:\Program Files\Blender Foundation\Blender 4.5\blender.exe" `
>>   -ArgumentList @("-b","C:\Users\marti\Desktop\tfg_teleco\proyectos\EsAppSingLenguageAI\pruebas_blender\cuerpo_humano_rigged.blend","--python","C:\Users\marti\Desktop\tfg_teleco\proyectos\EsAppSingLenguageAI\pruebas_blender\poses_library.py") `
>>   -NoNewWindow -Wait


'''
import bpy, json, os # type: ignore
import mathutils

# Helper minimal para manejar la carga de la librería de poses usando rutas relativas al script.
import json, os

def _detect_script_dir():
    try:
        # cuando se ejecuta como .py
        return os.path.dirname(os.path.abspath(__file__))
    except Exception:
        try:
            # cuando se ejecuta dentro de Blender Text Editor: usar carpeta del .blend
            return bpy.path.abspath("//")
        except Exception:
            return os.getcwd()

SCRIPT_DIR = _detect_script_dir()
DEFAULT_POSES_PATH = os.path.join(SCRIPT_DIR, "poses_library.json")

def load_poses(path=None):
    p = path or DEFAULT_POSES_PATH
    if not os.path.isabs(p):
        p = os.path.join(SCRIPT_DIR, p)
    if not os.path.exists(p):
        return {}
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)

def save_poses(data, path=None):
    p = path or DEFAULT_POSES_PATH
    if not os.path.isabs(p):
        p = os.path.join(SCRIPT_DIR, p)
    os.makedirs(os.path.dirname(p), exist_ok=True)
    with open(p, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

# ========= CONFIG =========
POSE_NAME = "B"
LIB_PATH  = bpy.path.abspath("//poses_library.json")
ONLY_PREFIX = ""    # si no quieres filtrar, déjalo vacío
INCLUDE_LOCATION = True
# ==================================

def active_armature():
    obj = bpy.context.object
    if obj and obj.type == "ARMATURE":
        return obj
    for o in bpy.context.selected_objects:
        if o.type == "ARMATURE":
            return o
    # fallback: primer armature en escena
    for o in bpy.data.objects:
        if o.type == "ARMATURE":
            return o
    return None

def bone_delta(pb):
    # Devuelve loc, rot_quat, scale en pose space (pose bone properties)
    # Se prefiere rotation_quaternion si está disponible
    loc = pb.location.copy() if hasattr(pb, "location") else mathutils.Vector((0.0,0.0,0.0))
    try:
        if pb.rotation_mode == 'QUATERNION':
            rot_quat = pb.rotation_quaternion.copy()
        else:
            # convertir rotation_euler a quaternion para consistencia
            rot_quat = pb.rotation_euler.to_quaternion()
    except Exception:
        rot_quat = mathutils.Quaternion((1.0,0.0,0.0,0.0))
    try:
        scl = pb.scale.copy()
    except Exception:
        scl = mathutils.Vector((1.0,1.0,1.0))
    return loc, rot_quat, scl

def is_identity(loc, quat, loc_tol=1e-6, rot_tol=1e-6):
    if loc.length > loc_tol:
        return False
    # quaternion close to identity (1,0,0,0)
    return abs(quat.w - 1.0) < rot_tol and abs(quat.x) < rot_tol and abs(quat.y) < rot_tol and abs(quat.z) < rot_tol

# ==== Construcción del diccionario de huesos: guardar TODOS los huesos ====
arm = active_armature()
if not arm:
    raise RuntimeError("No active armature found. Selecciona un armature o ejecútalo con uno abierto.")

# Info debug
print(f"[DEBUG] active armature: {arm.name}")
print(f"[DEBUG] bones in arm.data.bones: {len(arm.data.bones)}")
print(f"[DEBUG] bones in arm.pose.bones: {len(arm.pose.bones)}")
print(f"[DEBUG] LIB_PATH (abs): {os.path.abspath(LIB_PATH)}")

# Asegurar modo POSE
try:
    bpy.context.view_layer.objects.active = arm
    if bpy.context.mode != 'POSE':
        bpy.ops.object.mode_set(mode='POSE')
except Exception as e:
    print(f"[DEBUG] Warning setting pose mode: {e}")

bones_out = {}
kept = 0
missing_pose_bones = []

# iterar sobre todos los bones definidos en el armature (arm.data.bones)
for bone in arm.data.bones:
    name = bone.name
    if ONLY_PREFIX and not name.startswith(ONLY_PREFIX):
        continue

    try:
        pb = arm.pose.bones.get(name)
        if pb is not None:
            # pose matrix (en espacio del objeto armature)
            pose_mat = pb.matrix.copy()
            # posición en espacio del objeto y en espacio world
            loc = pose_mat.to_translation()
            world_loc = (arm.matrix_world @ pose_mat).to_translation()
            # rotación como quaternion (pose / world)
            rot_quat = pose_mat.to_quaternion()
            world_quat = (arm.matrix_world @ pose_mat).to_quaternion()
            # escala (pose bone scale, si existe)
            try:
                scl = pb.scale.copy()
            except Exception:
                scl = mathutils.Vector((1.0, 1.0, 1.0))
            # rot_euler preferible desde pb si existe
            try:
                rot_euler = [float(pb.rotation_euler.x), float(pb.rotation_euler.y), float(pb.rotation_euler.z)]
            except Exception:
                rot_euler = [float(rot_quat.to_euler().x), float(rot_quat.to_euler().y), float(rot_quat.to_euler().z)]
        else:
            # no hay canal de pose: usar rest pose (bone.head_local) en espacio del objeto
            rest_head = bone.head_local.copy()
            loc = rest_head
            world_loc = arm.matrix_world @ rest_head
            try:
                rot_quat = bone.matrix_local.to_quaternion()
            except Exception:
                rot_quat = mathutils.Quaternion((1.0, 0.0, 0.0, 0.0))
            world_quat = (arm.matrix_world @ bone.matrix_local).to_quaternion() if hasattr(bone, "matrix_local") else rot_quat
            rot_euler = [0.0, 0.0, 0.0]
            scl = mathutils.Vector((1.0, 1.0, 1.0))
    except Exception as e:
        print(f"[DEBUG] Error leyendo bone '{name}': {e}")
        loc = mathutils.Vector((0.0,0.0,0.0))
        world_loc = arm.matrix_world @ loc
        rot_quat = mathutils.Quaternion((1.0,0.0,0.0,0.0))
        world_quat = rot_quat
        rot_euler = [0.0,0.0,0.0]
        scl = mathutils.Vector((1.0,1.0,1.0))

    entry = {}
    if INCLUDE_LOCATION:
        entry["loc"] = [float(loc.x), float(loc.y), float(loc.z)]
        entry["world_loc"] = [float(world_loc.x), float(world_loc.y), float(world_loc.z)]
    entry["rot_quat"] = [float(rot_quat.w), float(rot_quat.x), float(rot_quat.y), float(rot_quat.z)]
    entry["world_rot_quat"] = [float(world_quat.w), float(world_quat.x), float(world_quat.y), float(world_quat.z)]
    entry["rot_euler"] = [float(rot_euler[0]), float(rot_euler[1]), float(rot_euler[2])]
    entry["scale"] = [float(scl.x), float(scl.y), float(scl.z)]

    bones_out[name] = entry
    kept += 1

# cargar/crear biblioteca y escribir
lib = {}
if os.path.exists(LIB_PATH):
    try:
        with open(LIB_PATH, "r", encoding="utf-8") as f:
            lib = json.load(f)
    except Exception:
        lib = {}

lib.setdefault("poses", {})[POSE_NAME] = {"bones": bones_out}
os.makedirs(os.path.dirname(LIB_PATH), exist_ok=True)
with open(LIB_PATH, "w", encoding="utf-8") as f:
    json.dump(lib, f, indent=2, ensure_ascii=False)

print(f"[OK] Pose '{POSE_NAME}' guardada/actualizada en {os.path.abspath(LIB_PATH)}. Huesos guardados: {kept}")
if missing_pose_bones:
    print(f"[DEBUG] Bones without pose channel (defaults used): {missing_pose_bones}")
