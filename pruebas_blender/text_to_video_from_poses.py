# text_to_video_from_poses.py
import bpy, json, os, sys, argparse, math

# Detectar directorio del script (compatible con ejecución dentro de Blender)
def _detect_script_dir():
    try:
        return os.path.dirname(os.path.abspath(__file__))
    except Exception:
        pass
    try:
        for txt in bpy.data.texts:
            fp = getattr(txt, "filepath", "")
            if fp:
                return os.path.dirname(bpy.path.abspath(fp))
    except Exception:
        pass
    try:
        blend_dir = bpy.path.abspath("//")
        if blend_dir:
            return blend_dir
    except Exception:
        pass
    return os.getcwd()

SCRIPT_DIR = _detect_script_dir()

# -------- CLI desde Blender --------
argv = sys.argv
argv = argv[argv.index("--")+1:] if "--" in argv else []
ap = argparse.ArgumentParser()
# dejar defaults como None y resolver a rutas relativas según SCRIPT_DIR después de parsear
ap.add_argument("--library", default=None)
ap.add_argument("--poses", default="")
ap.add_argument("--text", default="")
ap.add_argument("--armature", default="")
ap.add_argument("--fps", type=int, default=24)
ap.add_argument("--hold", type=int, default=12)
ap.add_argument("--transition", type=int, default=12)
ap.add_argument("--engine", choices=["EEVEE","CYCLES"], default="EEVEE")
ap.add_argument("--width", type=int, default=1080)
ap.add_argument("--height", type=int, default=1080)
ap.add_argument("--out", default=None)
ap.add_argument("--camera_lib", default=None)
ap.add_argument("--camera_name", default="Cam_01")
args = ap.parse_args(argv)

def log(m): print(f"[BLENDER] {m}")
def abspath(p):
    # si es una ruta Blender-style (//...) dejar que bpy.path.abspath la convierta
    try:
        if not p:
            return p
        # Blender-relative path
        if isinstance(p, str) and p.startswith("//"):
            return bpy.path.abspath(p)
    except Exception:
        pass
    # si no es absoluta, resolver respecto a SCRIPT_DIR
    if p and not os.path.isabs(p):
        return os.path.abspath(os.path.join(SCRIPT_DIR, p))
    return os.path.abspath(p) if p else p

def _load_camera_entry(path, name):
    """Carga camera_library.json y devuelve la entrada 'name' o None."""
    try:
        p = path or os.path.join(SCRIPT_DIR, "camera_library.json")
        p = abspath(p)
        if not os.path.exists(p):
            return None
        with open(p, "r", encoding="utf-8") as f:
            lib = json.load(f)
        cams = lib.get("cameras", {})
        return cams.get(name)
    except Exception as e:
        log(f"Warning: no se pudo leer camera lib: {e}")
        return None

def _apply_camera_entry(entry):
    if not entry:
        return
    try:
        log(f"Camera entry JSON: {json.dumps(entry, ensure_ascii=False, indent=2)}")
    except Exception:
        log(f"Camera entry: {entry}")

    obj_name = entry.get("object_name") or args.camera_name
    obj = bpy.data.objects.get(obj_name)
    if not obj or getattr(obj, "type", "") != "CAMERA":
        obj = bpy.data.objects.get(args.camera_name) or obj
    if not obj:
        cam_data = bpy.data.cameras.new(obj_name + "_data")
        obj = bpy.data.objects.new(obj_name, cam_data)
        bpy.context.collection.objects.link(obj)

    # transform (location + rotation)
    tr = entry.get("transform", {}) or {}
    loc = tr.get("location")
    if loc and len(loc) >= 3:
        try: obj.location = [float(loc[0]), float(loc[1]), float(loc[2])]
        except Exception: pass
    if "rotation_quaternion" in tr:
        try:
            obj.rotation_mode = 'QUATERNION'
            q = tr["rotation_quaternion"]
            obj.rotation_quaternion = [float(q[0]), float(q[1]), float(q[2]), float(q[3])]
        except Exception: pass
    elif "rotation_euler" in tr:
        try:
            rm = tr.get("rotation_mode", "XYZ")
            obj.rotation_mode = rm
            e = tr["rotation_euler"]
            obj.rotation_euler = [float(e[0]), float(e[1]), float(e[2])]
        except Exception: pass

    # camera data
    cdata = obj.data
    cinfo = entry.get("data", {}) or {}
    try:
        if "lens_mm" in cinfo:
            cdata.lens = float(cinfo["lens_mm"])
        if "sensor_width" in cinfo:
            cdata.sensor_width = float(cinfo["sensor_width"])
        if "sensor_height" in cinfo:
            cdata.sensor_height = float(cinfo["sensor_height"])
        if "shift_x" in cinfo:
            cdata.shift_x = float(cinfo["shift_x"])
        if "shift_y" in cinfo:
            cdata.shift_y = float(cinfo["shift_y"])
    except Exception:
        pass

    # DOF: focus_object preferred, else focus_distance; if none, fallback to armature distance
    try:
        dof = cdata.dof
        # aperture
        if "aperture_fstop" in cinfo and cinfo["aperture_fstop"] is not None:
            try: dof.aperture_fstop = float(cinfo["aperture_fstop"])
            except Exception: pass

        focus_name = cinfo.get("dof_focus_object")
        focus_dist = cinfo.get("dof_focus_distance")
        applied = False
        if focus_name:
            fo = bpy.data.objects.get(focus_name)
            if fo:
                try:
                    dof.focus_object = fo
                    dof.use_dof = True
                    applied = True
                except Exception: pass
        if not applied and focus_dist is not None:
            try:
                dof.focus_distance = float(focus_dist)
                dof.use_dof = True
                applied = True
            except Exception:
                applied = False
        # fallback: usar armature como objetivo si nada válido
        if not applied:
            target = None
            if args.armature:
                target = bpy.data.objects.get(args.armature)
            if target is None:
                for o in bpy.data.objects:
                    if o.type == 'ARMATURE':
                        target = o; break
            if target is not None:
                try:
                    cam_world = obj.matrix_world.to_translation()
                    tgt_world = target.matrix_world.to_translation()
                    dist = (cam_world - tgt_world).length
                    dof.focus_distance = float(dist)
                    dof.use_dof = True
                    log(f"DOF fallback focus_distance set to distance to armature: {dist:.3f}")
                except Exception:
                    pass
    except Exception as e:
        log(f"Warning DOF apply: {e}")

    # asegurar cámara de escena para render
    try:
        scn = bpy.context.scene
        scn.camera = obj
        log(f"Scene camera for render set to: {obj.name}")
    except Exception as e:
        log(f"Warning: no se pudo asignar la cámara a la escena: {e}")

    # Ajuste automático para evitar desenfoque excesivo:
    try:
        dof = obj.data.dof
        has_focus_obj = bool(cinfo.get("dof_focus_object"))
        has_focus_dist = cinfo.get("dof_focus_distance") is not None
        # Si no hay objetivo ni distancia válida, desactivar DOF
        if not has_focus_obj and not has_focus_dist:
            try:
                dof.use_dof = False
                log("DOF desactivado (sin objetivo ni distancia de enfoque).")
            except Exception:
                pass
        else:
            # si hay distancia, forzar fstop alto para menos desenfoque si no viene en JSON
            try:
                if not cinfo.get("aperture_fstop"):
                    dof.aperture_fstop = 8.0
                    log("Aperture fstop ajustado a 8.0 para reducir desenfoque.")
            except Exception:
                pass
    except Exception as e:
        log(f"Warning configurando DOF fallback: {e}")

def get_sequence():
    if args.poses:
        seq = [s.strip() for s in args.poses.split(",") if s.strip()]
    elif args.text:
        seq = [ch for ch in args.text if not ch.isspace()]
    else:
        raise RuntimeError("Indica --poses \"A,B\" o --text \"AB\"")
    return seq

def get_armature():
    if args.armature:
        ob = bpy.data.objects.get(args.armature)
        if ob and ob.type == 'ARMATURE':
            return ob
    ob = bpy.context.object
    if ob and ob.type == 'ARMATURE':
        return ob
    for o in bpy.context.selected_objects:
        if o.type == 'ARMATURE':
            return o
    raise RuntimeError("No se encontró un Armature. Pasa --armature o selecciónalo activo.")

def ensure_pose_mode(arm):
    bpy.context.view_layer.objects.active = arm
    if bpy.context.mode != 'POSE':
        bpy.ops.object.mode_set(mode='POSE')

def reset_pose(arm):
    for pb in arm.pose.bones:
        pb.matrix_basis.identity()

def apply_pose_dict(arm, bones_dict):
    for name, data in bones_dict.items():
        pb = arm.pose.bones.get(name)
        if not pb: 
            continue
        rq = data.get("rot_quat")
        re = data.get("rot_euler")
        if rq:
            pb.rotation_mode = 'QUATERNION'
            pb.rotation_quaternion = rq
        elif re:
            pb.rotation_mode = 'XYZ'
            pb.rotation_euler = re
        if "loc" in data:
            pb.location = data["loc"]

def insert_keys_for(arm, bones_dict, frame):
    for name in bones_dict.keys():
        pb = arm.pose.bones.get(name)
        if not pb: 
            continue
        if pb.rotation_mode == 'QUATERNION':
            pb.keyframe_insert(data_path="rotation_quaternion", frame=frame, group="POSE")
        else:
            pb.keyframe_insert(data_path="rotation_euler", frame=frame, group="POSE")
        if "loc" in bones_dict[name]:
            try:
                pb.keyframe_insert(data_path="location", frame=frame, group="POSE")
            except:
                pass

def set_interpolation(action, mode="BEZIER"):
    if not action: return
    for fc in action.fcurves:
        for kp in fc.keyframe_points:
            kp.interpolation = mode

# ---- Arranque
log("Cargando biblioteca de poses…")
lib_path = abspath(args.library)
if not os.path.exists(lib_path):
    raise FileNotFoundError(f"No existe: {lib_path}")
with open(lib_path, "r", encoding="utf-8") as f:
    lib = json.load(f)

poses_store = lib["poses"] if "poses" in lib else lib
sequence = get_sequence()
log(f"Secuencia: {sequence}")

arm = get_armature()
ensure_pose_mode(arm)

# AÑADIDO: referencia a la escena para evitar NameError (scn usada más abajo)
scn = bpy.context.scene

# aplicar cámara desde JSON si existe
cam_entry = _load_camera_entry(args.camera_lib, args.camera_name)
if cam_entry:
    _apply_camera_entry(cam_entry)

# Cámara y luz mínimas (por si falta)
if "Camera" not in bpy.data.objects:
    bpy.ops.object.camera_add(location=(0, -4.0, 1.7), rotation=(math.radians(85), 0, 0))
if not bpy.data.lights:
    bpy.ops.object.light_add(type='SUN', location=(3, -3, 5))

# Timeline
frame = 1
scn.frame_start = frame

for idx, pose_name in enumerate(sequence):
    log(f"Aplicando pose {pose_name} (paso {idx+1}/{len(sequence)})…")
    if "poses" in lib:
        entry = lib["poses"].get(pose_name)
        if not entry:
            raise KeyError(f"Pose '{pose_name}' no existe en {lib_path}")
        bones_dict = entry["bones"]
    else:
        bones_dict = lib["bones"]

    reset_pose(arm)
    apply_pose_dict(arm, bones_dict)
    insert_keys_for(arm, bones_dict, frame)  # inicio del hold

    frame_hold_end = frame + max(0, args.hold)
    if args.hold > 0:
        apply_pose_dict(arm, bones_dict)
        insert_keys_for(arm, bones_dict, frame_hold_end)

    frame = frame_hold_end + max(0, args.transition)

scn.frame_end = max(scn.frame_start + 1, frame)

if arm.animation_data and arm.animation_data.action:
    set_interpolation(arm.animation_data.action, "BEZIER")

# configurar render (engine, resolución, fps)
# scn.render.engine = args.engine
# -> reemplazar por intento robusto de asignar engine (Blender usa enums distintos según build)
engine_attempts = []
if str(args.engine).upper().startswith("EEVEE"):
    engine_attempts = ["BLENDER_EEVEE_NEXT", "BLENDER_EEVEE", "EEVEE"]
elif str(args.engine).upper().startswith("CYCLE"):
    engine_attempts = ["CYCLES", "CYCLE"]
else:
    engine_attempts = [args.engine]

set_engine_ok = False
for eng in engine_attempts:
    try:
        scn.render.engine = eng
        log(f"Engine de render establecido: {eng}")
        set_engine_ok = True
        break
    except Exception:
        continue

if not set_engine_ok:
    log(f"Aviso: no se pudo establecer el engine solicitado '{args.engine}'. Usando valor por defecto: {scn.render.engine}")

scn.render.resolution_x = args.width
scn.render.resolution_y = args.height
scn.render.fps = args.fps

# establecer ruta de salida absoluta y asegurar carpeta
out_path = abspath(args.out)
out_dir = os.path.dirname(out_path) or os.getcwd()
os.makedirs(out_dir, exist_ok=True)
scn.render.filepath = out_path

# FORZAR salida como vídeo (FFMPEG) — evita que Blender escriba una secuencia de imágenes
scn.render.image_settings.file_format = 'FFMPEG'
scn.render.image_settings.color_mode = 'RGB'
try:
    ff = scn.render.ffmpeg
    ff.format = 'MPEG4'
    ff.codec = 'H264'
    if hasattr(ff, "audio_codec"):
        ff.audio_codec = 'AAC'
    if hasattr(ff, "gopsize"):
        ff.gopsize = max(1, args.fps)
    if hasattr(ff, "constant_rate_factor"):
        ff.constant_rate_factor = 'MEDIUM'
    # distintos nombres de la propiedad pixel format en builds diferentes
    for pix_attr in ("pix_fmt", "pixfmt", "pixel_format"):
        if hasattr(ff, pix_attr):
            try:
                setattr(ff, pix_attr, 'YUV420P')
                break
            except Exception:
                pass
except Exception as e:
    log(f"Aviso: no se pudo configurar FFmpeg: {e}")

log(f"Preparado para render. Rango: {scn.frame_start}..{scn.frame_end}")
log(f"Salida: {out_path}")

# lanzar render
try:
    bpy.ops.render.render(animation=True)
except Exception as e:
    log(f"Error durante el render: {e}")
    import traceback
    print(traceback.format_exc())
    sys.exit(1)

# Comprobación final: asegurar que se generó un .mp4 válido
if not os.path.exists(out_path) or os.path.getsize(out_path) == 0:
    log(f"Error: no se generó el vídeo en {out_path}. Se ha creado posiblemente una secuencia de imágenes en {out_dir}.")
    # mostrar ejemplo de archivos en la carpeta para depuración
    try:
        sample = os.listdir(out_dir)[:10]
        log(f"Contenido de {out_dir} (muestra): {sample}")
    except Exception:
        pass
    sys.exit(1)

log("Render finalizado correctamente.")
