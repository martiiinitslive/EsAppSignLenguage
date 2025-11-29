# ============================================
# STEP 1: Capture MediaPipe data
# ============================================
import cv2
import json
import mediapipe as mp
from pathlib import Path
import argparse
import datetime

# base app-back directory (relative pathing)
BASE_DIR = Path(__file__).resolve().parent.parent.parent
DEFAULT_INPUT_DIR = BASE_DIR / "data" / "dataset-en-bruto" / "asl_dataset" / "mediapipe_poses"

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

def capture_hand_pose_to_json(image_path, gesture_name=None, convert_to_blender=False, write_json=False, out_path=None):
    """
    Process an image and return a dict with the landmarks.
    - If write_json and out_path are provided, also write an individual file (optional).
    """
    img_p = Path(image_path)
    if not img_p.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")
    image = cv2.imread(str(img_p))
    if image is None:
        raise RuntimeError(f"OpenCV could not read the image (empty): {image_path}")

    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    with mp_hands.Hands(static_image_mode=True, max_num_hands=1, min_detection_confidence=0.5) as hands:
        results = hands.process(image_rgb)

    out = {
        "source_image": str(img_p),
        "gesture_name": gesture_name,
        "hand_detected": False,
        "landmarks": None,
        "world_landmarks": None,
        "handedness": None
    }

    if not results.multi_hand_landmarks:
        if write_json and out_path:
            Path(out_path).parent.mkdir(parents=True, exist_ok=True)
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(out, f, indent=2, ensure_ascii=False)
        return out  # no hand detected

    # use first detected hand
    hand_landmarks = results.multi_hand_landmarks[0]
    handedness = None
    if results.multi_handedness:
        handedness = results.multi_handedness[0].classification[0].label

    norm_lms = [{"x": float(lm.x), "y": float(lm.y), "z": float(lm.z)} for lm in hand_landmarks.landmark]

    world_lms = None
    if getattr(results, "multi_hand_world_landmarks", None):
        try:
            w = results.multi_hand_world_landmarks[0]
            world_lms = [{"x": float(lm.x), "y": float(lm.y), "z": float(lm.z)} for lm in w.landmark]
        except Exception:
            world_lms = None

    if convert_to_blender and world_lms:
        world_lms_blender = []
        for p in world_lms:
            world_lms_blender.append({"x": float(p["x"]), "y": float(-p["z"]), "z": float(p["y"])})
        world_lms = world_lms_blender

    out.update({
        "hand_detected": True,
        "handedness": handedness,
        "landmarks": norm_lms,
        "world_landmarks": world_lms
    })

    if write_json and out_path:
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(out, f, indent=2, ensure_ascii=False)

    return out

def _iter_images_in_dir(d):
    p = Path(d)
    for ext in ("*.png","*.jpg","*.jpeg","*.bmp","*.tiff"):
        for f in sorted(p.glob(ext)):
            yield f

def main():
    ap = argparse.ArgumentParser(description="Capture MediaPipe hand landmarks from image(s) -> single JSON")
    ap.add_argument("--image", help="Path to a single image")
    ap.add_argument("--input_dir", default=str(DEFAULT_INPUT_DIR),
                    help=f"Folder with images (processes all). Default: {DEFAULT_INPUT_DIR}")
    ap.add_argument("--out_dir", default="./hand_poses_json", help="Output folder for individual JSONs (optional)")
    ap.add_argument("--single_output", default=str(BASE_DIR / "mp" / "output_mp" / (Path(__file__).stem + ".json")),
                    help="Single JSON file where all poses will be saved (default: app-back/mp/output_mp)")
    ap.add_argument("--gesture", default=None, help="Gesture/pose name (optional) for all images")
    ap.add_argument("--blender", action="store_true", help="Convert world_landmarks to Blender-style coordinates (x,-z,y)")
    ap.add_argument("--write_individual", action="store_true", help="Also write individual JSON per image to --out_dir")
    args = ap.parse_args()

    if not args.image and not args.input_dir:
        raise RuntimeError("Pass --image <path> or --input_dir <folder>")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    jobs = []
    if args.image:
        jobs.append(Path(args.image))
    if args.input_dir:
        jobs.extend(list(_iter_images_in_dir(args.input_dir)))

    if not jobs:
        print("No images found to process.")
        return

    all_poses = {}
    meta = {
        "generated_at": datetime.datetime.utcnow().isoformat() + "Z",
        "source_count": len(jobs),
        "converted_to_blender": bool(args.blender)
    }

    for img in jobs:
        try:
            # derive gesture name
            if args.gesture:
                gesture_name = args.gesture
            else:
                stem = img.stem.strip()
                gesture_name = stem.split("_")[0].strip().upper() if stem else img.stem

            individual_out = out_dir / (img.stem + ".json") if args.write_individual else None
            result = capture_hand_pose_to_json(str(img), gesture_name=gesture_name,
                                               convert_to_blender=args.blender,
                                               write_json=bool(individual_out), out_path=str(individual_out) if individual_out else None)
            key = gesture_name or img.stem
            # handle duplicates: store list if multiple images for same gesture
            if key in all_poses:
                # convert single entry to list
                if isinstance(all_poses[key], list):
                    all_poses[key].append(result)
                else:
                    all_poses[key] = [all_poses[key], result]
            else:
                all_poses[key] = result

            if result.get("hand_detected"):
                print(f"Processed: {img} -> gesture '{gesture_name}'")
            else:
                print(f"No hand detected in: {img} (gesture '{gesture_name}')")
        except Exception as e:
            print(f"Error processing {img}: {e}")

    # write single JSON
    single_out_p = Path(args.single_output)
    single_out_p.parent.mkdir(parents=True, exist_ok=True)
    final = {
        "meta": meta,
        "poses": all_poses
    }
    with open(single_out_p, "w", encoding="utf-8") as f:
        json.dump(final, f, indent=2, ensure_ascii=False)

    print(f"✅ Saved {len(all_poses)} poses to: {single_out_p}")

if __name__ == "__main__":
    main()
