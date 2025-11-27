"""
Apply MediaPipe landmark-based bones + angles to a Blender armature.

Usage (from system / PowerShell):
& "C:\Program Files\Blender Foundation\Blender 4.5\blender.exe" "path\to\your.blend" --background --python "apply_angles_to_rig.py" -- --angles "info_angles_video.json" --landmarks "poses_mediapipe_video.json" --armature "ArmatureName" --mode apply --start-frame 1 --frame-step 1

Modes:
  --mode dry-run        : just print mapping suggestions
  --mode helper         : create/update a helper armature with bones for each MP connection (visual)
  --mode apply          : apply rotations to rig bones (requires mapping)

Important:
 - The script expects "landmarks" JSON to contain world_landmarks or landmarks as lists/dicts.
 - Provide a BONE_MAP mapping landmark_index -> target bone name in your rig (below) or use --auto-map.
"""
import sys, json, math, argparse
from pathlib import Path

try:
    import bpy
    from mathutils import Vector, Matrix, Quaternion
except Exception:
    raise SystemExit("Run this script inside Blender (bpy required).")

# --- CONFIG: default mapping index -> bone name (edit to your rig) ---
# MediaPipe hand indices -> bones in Human.rig (left hand)
# We map each landmark index to the bone whose tail should end at that landmark
# (e.g. 0->1 uses bone mapped at index 1). Indices without a matching bone are ignored.
BONE_MAP = {
    # Wrist to thumb base
    "0": "wrist.L",        # head of wrist
    "1": "wrist.L",        # tail of wrist
    # Thumb (1..4)
    "2": "finger1-1.L",
    "3": "finger1-2.L",
    "4": "finger1-3.L",
    # Index (5..8)
    "6": "finger2-1.L",
    "7": "finger2-2.L",
    "8": "finger2-3.L",
    # Middle (9..12)
    "10": "finger3-1.L",
    "11": "finger3-2.L",
    "12": "finger3-3.L",
    # Ring (13..16)
    "14": "finger4-1.L",
    "15": "finger4-2.L",
    "16": "finger4-3.L",
    # Pinky (17..20)
    "18": "finger5-1.L",
    "19": "finger5-2.L",
    "20": "finger5-3.L",
}
# --------------------------------------------------------------------

# MediaPipe hand connections (pairs of landmark indices)
HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (17, 18), (18, 19), (19, 20),
    (0, 17), (0, 13), (0, 9), (0, 5)
]

def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def to_vec3(pt):
    """Normalize landmark representation to Vector (x,y,z).
    Accept dict {'x','y','z'} or list/tuple [x,y,z] or [x,y].
    If values look normalized (0..1) user may want to scale/translate later.
    """
    if pt is None:
        return None
    if isinstance(pt, dict):
        x = pt.get("x"); y = pt.get("y"); z = pt.get("z", 0.0)
        if x is None or y is None:
            return None
        return Vector((float(x), float(y), float(z if z is not None else 0.0)))
    if isinstance(pt, (list, tuple)):
        if len(pt) >= 2:
            x = float(pt[0]); y = float(pt[1]); z = float(pt[2]) if len(pt) > 2 else 0.0
            return Vector((x, y, z))
    return None

def build_neighbor_map(connections=HAND_CONNECTIONS):
    from collections import defaultdict
    nbr = defaultdict(set)
    for a,b in connections:
        nbr[a].add(b); nbr[b].add(a)
    return {k: sorted(v) for k,v in nbr.items()}


def extract_landmarks_frame(landmarks_data, pose_name, frame_index):
    """Return a list of landmarks for a given pose/frame from several possible shapes."""
    if not landmarks_data:
        return []
    poses = landmarks_data.get("poses") if isinstance(landmarks_data, dict) else None
    pose_entry = poses.get(pose_name) if poses and pose_name in poses else None

    # Fallback: if the JSON is already a list of frames
    if pose_entry is None and isinstance(landmarks_data, list):
        pose_entry = landmarks_data

    if pose_entry is None:
        return []

    # pick frame
    frame = None
    if isinstance(pose_entry, list):
        if frame_index < len(pose_entry):
            frame = pose_entry[frame_index]
    elif isinstance(pose_entry, dict):
        frames = pose_entry.get("frames", []) or []
        if frame_index < len(frames):
            frame = frames[frame_index]
    if frame is None:
        return []

    # normalize frame to a list of points
    if isinstance(frame, list):
        return frame
    if isinstance(frame, dict):
        for key in ("world_landmarks", "landmarks", "multi_hand_landmarks", "hand_landmarks", "pose_landmarks"):
            if key in frame and frame[key]:
                return frame[key]
        if all(isinstance(k, str) and k.isdigit() for k in frame.keys()):
            return [frame[k] for k in sorted(frame.keys(), key=lambda x: int(x))]
    return []

# ---- Helper: create/update visual helper armature ----
def get_or_create_helper_armature(name="MP_Helper"):
    obj = bpy.data.objects.get(name)
    if obj and obj.type == 'ARMATURE':
        return obj
    # create new armature object
    arm = bpy.data.armatures.new(name + ".data")
    obj = bpy.data.objects.new(name, arm)
    bpy.context.collection.objects.link(obj)
    return obj

def update_helper_armature_from_landmarks(helper_obj, landmarks, connections=HAND_CONNECTIONS, convert_blender_axis=False):
    """Create or update bones for each connection using landmark positions.
    convert_blender_axis: if True, convert MP world->Blender by (x,-z,y) — set according to your capture.
    """
    if helper_obj.mode != 'OBJECT':
        bpy.ops.object.mode_set(mode='OBJECT')
    bpy.context.view_layer.objects.active = helper_obj
    arm = helper_obj.data
    # Switch to edit mode to create/edit bones
    bpy.ops.object.mode_set(mode='EDIT')
    edit_bones = arm.edit_bones
    for a,b in connections:
        name = f"mp_{a}_{b}"
        eb = edit_bones.get(name)
        v_a = to_vec3(landmarks[a]) if a < len(landmarks) else None
        v_b = to_vec3(landmarks[b]) if b < len(landmarks) else None
        if v_a is None or v_b is None:
            continue
        if convert_blender_axis:
            v_a = Vector((v_a.x, -v_a.z, v_a.y))
            v_b = Vector((v_b.x, -v_b.z, v_b.y))
        if eb is None:
            eb = edit_bones.new(name)
        # set head/tail in armature local space — assume landmarks are roughly in armature/world space
        eb.head = v_a
        eb.tail = v_b
        eb.use_connect = False
    bpy.ops.object.mode_set(mode='OBJECT')


# ------------------------------------------------------------------
# Function: apply a position_A JSON to an armature using landmarks
# - position_a_path: JSON produced by `generate_position_A.py` (landmarks_converted + mapped_bones)
# - angles_json_path: optional path to `info_angles_video.json` (angles per pose/frame)
# - pose_name/frame_idx: optional selection for angles JSON
# - keyframe: if True, insert keyframes after applying pose
def apply_pose_from_positionA(arm_obj, position_a_path, angles_json_path=None, pose_name=None, frame_idx=0, keyframe=False):
    import json
    if arm_obj is None:
        print(f"[ERROR] apply_pose_from_positionA: armature is None")
        return False

    try:
        with open(position_a_path, 'r', encoding='utf-8') as f:
            pos = json.load(f)
    except Exception as e:
        print(f"[ERROR] cannot read position A: {e}")
        return False

    landmarks = pos.get('landmarks_converted') or []
    mapped = pos.get('mapped_bones', {})
    if not landmarks:
        print('[ERROR] position A contains no landmarks')
        return False

    # optional load angles
    angles_frame = None
    if angles_json_path:
        try:
            with open(angles_json_path, 'r', encoding='utf-8') as f:
                angles_js = json.load(f)
            poses = angles_js.get('poses', {})
            if pose_name and pose_name in poses:
                pinfo = poses[pose_name]
                frames = pinfo.get('frames', [])
                if frames and frame_idx < len(frames):
                    angles_frame = frames[frame_idx]
            else:
                # try first pose available
                for k, v in poses.items():
                    frames = v.get('frames', [])
                    if frames:
                        angles_frame = frames[min(frame_idx, len(frames)-1)]
                        break
        except Exception as e:
            print('[WARN] Could not load angles JSON:', e)

    # helper to get vector between two landmark indices
    def lm_vec(a_idx, b_idx):
        try:
            a = landmarks[int(a_idx)]
            b = landmarks[int(b_idx)]
            return Vector((b[0]-a[0], b[1]-a[1], b[2]-a[2]))
        except Exception:
            return None

    # operate in pose mode
    bpy.context.view_layer.objects.active = arm_obj
    bpy.ops.object.mode_set(mode='POSE')

    # For each mapped index -> bone, try to orient bone using the connection where index is the second element (a->b)
    for idx_str, info in mapped.items():
        bone_name = info.get('bone_name')
        if not bone_name:
            continue
        idx = int(idx_str)
        target_vec = None
        used_neighbor = None
        # prefer connections where idx is the 'b' (end) as in other functions
        for a, b in HAND_CONNECTIONS:
            if b == idx:
                target_vec = lm_vec(a, b)
                used_neighbor = a
                break
        # if not found, try reversed where idx is the 'a'
        if target_vec is None:
            for a, b in HAND_CONNECTIONS:
                if a == idx:
                    target_vec = lm_vec(a, b)
                    used_neighbor = b
                    # reverse direction for alignment to point from idx->neighbor
                    if target_vec is not None:
                        target_vec = -target_vec
                    break
        if target_vec is None:
            continue

        pbone = arm_obj.pose.bones.get(bone_name)
        if pbone is None:
            print(f"[WARN] Bone '{bone_name}' not found in armature; skipping")
            continue
        try:
            ok = align_pose_bone_to_vector(pbone, target_vec, arm_obj)
            if not ok:
                print(f"[WARN] align failed for bone {bone_name}")
                continue

            # apply angle from angles_frame if available: look up joint idx
            angle_val = None
            if isinstance(angles_frame, dict):
                jinfo = angles_frame.get(str(idx))
                if isinstance(jinfo, dict):
                    for pr in jinfo.get("pairs", []):
                        neigh = pr.get("neighbors", [])
                        if len(neigh) == 2 and used_neighbor is not None and (neigh[0] == used_neighbor or neigh[1] == used_neighbor):
                            angle_val = pr.get("angle")
                            if angle_val is not None:
                                try:
                                    angle_val = float(angle_val)
                                except Exception:
                                    angle_val = None
                                break
            if angle_val is not None:
                # apply angle around local axis X by default (match other function)
                if pbone.rotation_mode != 'XYZ':
                    pbone.rotation_mode = 'XYZ'
                e = pbone.rotation_euler
                # assume X axis is flexion; use positive angle as measured
                e.x += math.radians(angle_val)
                pbone.rotation_euler = e

            if keyframe:
                if pbone.rotation_mode == 'QUATERNION':
                    pbone.keyframe_insert(data_path='rotation_quaternion', frame=bpy.context.scene.frame_current)
                else:
                    pbone.keyframe_insert(data_path='rotation_euler', frame=bpy.context.scene.frame_current)
        except Exception as e:
            print(f"[WARN] failed to orient/apply angle to bone {bone_name}: {e}")

    bpy.ops.object.mode_set(mode='OBJECT')
    print('[INFO] apply_pose_from_positionA: applied landmarks to armature (angles may have been ignored)')
    return True
    return helper_obj

# ---- Core: apply orientations + angles to target rig ----
def landmark_to_world(landmark_vec, convert_blender_axis=False, scale=1.0, offset=Vector((0,0,0))):
    """Convert landmark vec to world coordinates for your scene.
    If landmarks are normalized (0..1) you might supply scale/offset to place them near armature.
    convert_blender_axis: transform (x,y,z)->(x,-z,y) if world_landmarks were converted differently.
    """
    if landmark_vec is None:
        return None
    v = Vector((landmark_vec.x, landmark_vec.y, landmark_vec.z))
    if convert_blender_axis:
        v = Vector((v.x, -v.z, v.y))
    return (v * scale) + offset

def align_pose_bone_to_vector(pbone, target_vec_world, arm_obj):
    """Align pose bone's local +Y (Blender convention for bones head->tail) to target_vec_world.
    This sets pbone.rotation_quaternion to rotate current bone vector to target vector in armature/world space.
    """
    # bone's rest head/tail in armature local space:
    bone = pbone.bone
    # current bone vector in armature (edit) space:
    head_local = bone.head_local
    tail_local = bone.tail_local
    cur_vec_local = (tail_local - head_local)
    if cur_vec_local.length < 1e-8 or target_vec_world.length < 1e-8:
        return False
    # convert target_vec_world to armature local
    arm_inv = arm_obj.matrix_world.inverted()
    target_local = arm_inv.to_3x3() @ target_vec_world
    # compute rotation that rotates cur_vec_local -> target_local
    try:
        rot = cur_vec_local.normalized().rotation_difference(target_local.normalized())
    except Exception:
        return False
    # Apply rotation to pose bone using quaternions
    pbone.rotation_mode = 'QUATERNION'
    pbone.rotation_quaternion = rot
    return True

def apply_angles_and_alignment_for_frame(arm_obj, final_map, landmarks_frame, angles_frame, convert_blender_axis=False, scale=1.0, offset=Vector((0,0,0)), axis_for_angle='X', angle_sign=1.0):
    """
    Apply alignment+angle to target armature for a single frame.
    - final_map: dict index_str -> bone_name (landmark index -> bone that ends at that landmark)
      We assume the bone that should align with connection (a->b) is the bone mapped at index b.
    - landmarks_frame: list of landmarks (dicts/lists)
    - angles_frame: dict from obtain_angles_video.py for that frame (joint index str -> {"pairs":[...]} )
    """
    # convert input landmarks to world vectors
    pts = [to_vec3(p) for p in landmarks_frame] if landmarks_frame else []
    world_pts = []
    for v in pts:
        if v is None:
            world_pts.append(None)
        else:
            world_pts.append(landmark_to_world(v, convert_blender_axis, scale, offset))

    # reset target bones so animations do not accumulate frame-to-frame
    targets = {bn for bn in final_map.values() if bn}
    bpy.context.view_layer.objects.active = arm_obj
    bpy.ops.object.mode_set(mode='POSE')
    for bname in targets:
        pb = arm_obj.pose.bones.get(bname)
        if pb:
            pb.rotation_mode = 'QUATERNION'
            pb.rotation_quaternion = Quaternion((1.0, 0.0, 0.0, 0.0))

    # operate in pose mode
    bpy.context.view_layer.objects.active = arm_obj
    bpy.ops.object.mode_set(mode='POSE')

    for (a,b) in HAND_CONNECTIONS:
        if b >= len(world_pts) or a >= len(world_pts):
            continue
        pa = world_pts[a]; pb = world_pts[b]
        if pa is None or pb is None:
            continue
        bone_name = final_map.get(str(b))
        if not bone_name:
            continue
        pbone = arm_obj.pose.bones.get(bone_name)
        if pbone is None:
            print(f"[WARN] Bone '{bone_name}' not found in armature; skipping connection {a}->{b}")
            continue
        targ_vec = (pb - pa)
        # align bone to vector
        ok = align_pose_bone_to_vector(pbone, targ_vec, arm_obj)
        if not ok:
            continue
        # apply additional scalar angle if available in angles_frame for joint b and neighbors pair including a
        angle_val = None
        if isinstance(angles_frame, dict):
            jinfo = angles_frame.get(str(b))
            if isinstance(jinfo, dict):
                for pr in jinfo.get("pairs", []):
                    neigh = pr.get("neighbors", [])
                    if len(neigh) == 2 and ((neigh[0] == a and neigh[1] == b) or (neigh[1] == a and neigh[0] == b) or neigh[0]==a or neigh[1]==a):
                        # Note: angle computed at joint b between neighbors; heuristically use it
                        angle_val = pr.get("angle")
                        if angle_val is not None:
                            angle_val = float(angle_val) * angle_sign
                            break
        if angle_val is not None:
            # rotate around bone local axis (choose axis_for_angle)
            # apply small rotation in Euler local space
            if pbone.rotation_mode != 'XYZ':
                pbone.rotation_mode = 'XYZ'
            e = pbone.rotation_euler
            if axis_for_angle.upper() == 'X':
                e.x += math.radians(angle_val)
            elif axis_for_angle.upper() == 'Y':
                e.y += math.radians(angle_val)
            else:
                e.z += math.radians(angle_val)
            pbone.rotation_euler = e

    bpy.ops.object.mode_set(mode='OBJECT')

# ---- Auto-map heuristic (try to match bone names in armature) ----
def find_bones_with_keyword(arm_obj, keyword):
    kw = keyword.lower()
    return [b.name for b in arm_obj.data.bones if kw in b.name.lower()]

def auto_map_indices_to_bones(arm_obj):
    groups = {
        "thumb": list(range(1,5)),
        "index": list(range(5,9)),
        "middle": list(range(9,13)),
        "ring": list(range(13,17)),
        "pinky": list(range(17,21)),
        "wrist": [0],
    }
    result = {}
    for key, indices in groups.items():
        found = find_bones_with_keyword(arm_obj, key)
        if not found:
            continue
        found_sorted = sorted(found)
        for i, idx in enumerate(indices):
            if i < len(found_sorted):
                result[str(idx)] = found_sorted[i]
    return result

# ----------------- CLI entrypoint -----------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--angles", required=True, help="info_angles_video.json")
    ap.add_argument("--landmarks", required=False, help="poses_mediapipe_video.json (optional, for positions)")
    ap.add_argument("--armature", required=True, help="Target rig armature object name")
    ap.add_argument("--mode", choices=("dry-run","helper","apply"), default="dry-run", help="Operation mode")
    ap.add_argument("--start-frame", type=int, default=1)
    ap.add_argument("--frame-step", type=int, default=1)
    ap.add_argument("--auto-map", action="store_true", help="Try to auto-map indices->bones by name heuristics")
    ap.add_argument("--create-helper", action="store_true", help="Create/update helper armature visualizing MP connections")
    ap.add_argument("--convert-blender-axis", action="store_true", help="Convert landmarks using (x,-z,y) convention")
    ap.add_argument("--scale", type=float, default=1.0, help="Scale landmarks to scene units")
    ap.add_argument("--offset-x", type=float, default=0.0)
    ap.add_argument("--offset-y", type=float, default=0.0)
    ap.add_argument("--offset-z", type=float, default=0.0)
    ap.add_argument("--axis", choices=("X","Y","Z"), default="X", help="Axis to apply the scalar angle around")
    ap.add_argument("--sign", type=float, default=1.0, help="Sign multiplier for scalar angle")
    args, unknown = ap.parse_known_args(sys.argv[sys.argv.index("--")+1:] if "--" in sys.argv else [])

    angles_p = Path(args.angles)
    if not angles_p.exists():
        raise SystemExit(f"Angles JSON not found: {angles_p}")
    angles_data = load_json(angles_p)
    poses_angles = angles_data.get("poses", {})

    landmarks_data = None
    if args.landmarks:
        lp = Path(args.landmarks)
        if lp.exists():
            landmarks_data = load_json(lp)
        else:
            print("[WARN] landmarks JSON not found; only angles will be used.")

    arm_obj = bpy.data.objects.get(args.armature)
    if arm_obj is None or arm_obj.type != 'ARMATURE':
        raise SystemExit(f"Armature '{args.armature}' not found or not an armature in the current .blend")

    # decide mapping
    final_map = dict(BONE_MAP)
    if args.auto_map:
        am = auto_map_indices_to_bones(arm_obj)
        for k,v in am.items():
            final_map.setdefault(k, v)

    print(f"[INFO] Using mapping entries: {len(final_map)}; auto_map={'on' if args.auto_map else 'off'}")

    # helper armature creation (visual)
    helper = None
    if args.create_helper or args.mode == "helper":
        helper = get_or_create_helper_armature("MP_Helper")
        # We'll update helper using landmarks from the first pose/frame available (if any)
        if landmarks_data:
            # choose first pose/frame with landmarks
            any_pose = next(iter(landmarks_data.get("poses", {}).values()), None)
            if any_pose:
                frames = any_pose.get("frames", []) or []
                if frames:
                    first_frame = frames[0]
                    lms = first_frame.get("world_landmarks") or first_frame.get("landmarks") or []
                    update_helper_armature_from_landmarks(helper, lms, HAND_CONNECTIONS, convert_blender_axis=args.convert_blender_axis)
                    print("[INFO] Helper armature updated from first frame landmarks.")
        if args.mode == "helper" and not args.create_helper:
            print("[INFO] Helper created/updated; exiting (mode=helper).")
            return

    if args.mode == "dry-run":
        print("[DRY RUN] Mapping preview:")
        for k in sorted(final_map.keys(), key=lambda x:int(x)):
            print(f"  idx {k} -> {final_map[k]}")
        print("[DRY RUN] Done.")
        return

    # mode == apply: iterate poses/frames and apply
    frame_cursor = args.start_frame
    scale = args.scale
    offset = Vector((args.offset_x, args.offset_y, args.offset_z))

    for pose_name, pinfo in poses_angles.items():
        frames = pinfo.get("frames", []) or []
        for fi, fentry in enumerate(frames):
            if (fi % args.frame_step) != 0:
                continue
            angles_frame = fentry.get("angles") or {}
            # load corresponding landmarks frame if provided (try same letter & frame_index)
            lframe = None
            if landmarks_data:
                lpose = landmarks_data.get("poses", {}).get(pose_name)
                if lpose:
                    lframes = lpose.get("frames", []) or []
                    if fi < len(lframes):
                        lframe = lframes[fi].get("world_landmarks") or lframes[fi].get("landmarks") or []
            # apply for this frame
            try:
                apply_angles_and_alignment_for_frame(
                    arm_obj,
                    final_map,
                    lframe or [],
                    angles_frame,
                    convert_blender_axis=args.convert_blender_axis,
                    scale=scale,
                    offset=offset,
                    axis_for_angle=args.axis,
                    angle_sign=args.sign
                )
                # write keyframes for rotated bones
                bpy.ops.object.mode_set(mode='POSE')
                for idx_str, bone_name in final_map.items():
                    pb = arm_obj.pose.bones.get(bone_name)
                    if not pb:
                        continue
                    # ensure Euler/quaternion inserted consistent with rotation_mode
                    if pb.rotation_mode == 'QUATERNION':
                        pb.keyframe_insert(data_path="rotation_quaternion", frame=frame_cursor)
                    else:
                        pb.keyframe_insert(data_path="rotation_euler", frame=frame_cursor)
                bpy.ops.object.mode_set(mode='OBJECT')
            except Exception as e:
                print(f"[ERROR] applying frame {fi} of pose {pose_name}: {e}")
            frame_cursor += 1

    print(f"Applied frames and keyframed up to frame {frame_cursor-1}")

if __name__ == "__main__":
    main()
