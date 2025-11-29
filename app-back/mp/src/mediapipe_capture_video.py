# ============================================
# STEP 1: Capture MediaPipe data FROM VIDEOS
# ============================================
import cv2
import json
import mediapipe as mp
from pathlib import Path
import argparse
import datetime
import numpy as np

# base app-back directory (relative pathing)
BASE_DIR = Path(__file__).resolve().parent.parent.parent
DEFAULT_INPUT_DIR = BASE_DIR / "data" / "dataset-en-bruto" / "asl_dataset" / "videos" / "videos_de_las_letras"

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

def extract_hand_landmarks_from_video(video_path, gesture_name=None, convert_to_blender=False, 
                                      sample_every_n_frames=1, max_frames=None):
    """
    Process a video and extract MediaPipe landmarks per frame.

    Args:
        video_path: path to the video file
        gesture_name: gesture identifier (e.g. "A", "B", "C")
        convert_to_blender: if True, convert world_landmarks to Blender axes
        sample_every_n_frames: sample every N frames (speed up)
        max_frames: maximum frames to process per video (None = all)

    Returns:
        dict with processed video structure
    """
    video_p = Path(video_path)
    if not video_p.exists():
        raise FileNotFoundError(f"Video not found: {video_path}")
    
    cap = cv2.VideoCapture(str(video_p))
    if not cap.isOpened():
        raise RuntimeError(f"OpenCV could not open the video: {video_path}")
    
    # Get video info
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    print(f"  Processing: {video_p.name}")
    print(f"    FPS: {fps}, Total frames: {total_frames}, Resolution: {width}x{height}")
    
    frame_landmarks = []
    frames_processed = 0
    frame_count = 0
    hand_detected_count = 0
    
    with mp_hands.Hands(static_image_mode=False, max_num_hands=1, 
                       min_detection_confidence=0.5, min_tracking_confidence=0.5) as hands:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_count += 1
            
            # Skip frames according to sample_every_n_frames
            if frame_count % sample_every_n_frames != 0:
                continue
            
            if max_frames and frames_processed >= max_frames:
                break
            
            image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(image_rgb)
            
            frame_data = {
                "frame_number": frame_count,
                "timestamp_sec": frame_count / fps if fps > 0 else 0,
                "hand_detected": False,
                "handedness": None,
                "landmarks": None,
                "world_landmarks": None
            }
            
            if results.multi_hand_landmarks:
                hand_landmarks = results.multi_hand_landmarks[0]
                handedness = None
                if results.multi_handedness:
                    handedness = results.multi_handedness[0].classification[0].label
                
                # Normalized landmarks (0-1)
                norm_lms = [
                    {"x": float(lm.x), "y": float(lm.y), "z": float(lm.z)} 
                    for lm in hand_landmarks.landmark
                ]
                
                # World landmarks (real-world coordinates in meters)
                world_lms = None
                if hasattr(results, "multi_hand_world_landmarks") and results.multi_hand_world_landmarks:
                    try:
                        w = results.multi_hand_world_landmarks[0]
                        world_lms = [
                            {"x": float(lm.x), "y": float(lm.y), "z": float(lm.z)} 
                            for lm in w.landmark
                        ]
                    except Exception:
                        world_lms = None
                
                # Convert to Blender axes if requested
                if convert_to_blender and world_lms:
                    world_lms_blender = []
                    for p in world_lms:
                        world_lms_blender.append({
                            "x": float(p["x"]), 
                            "y": float(-p["z"]), 
                            "z": float(p["y"])
                        })
                    world_lms = world_lms_blender
                
                frame_data.update({
                    "hand_detected": True,
                    "handedness": handedness,
                    "landmarks": norm_lms,
                    "world_landmarks": world_lms
                })
                
                hand_detected_count += 1
            
            frame_landmarks.append(frame_data)
            frames_processed += 1
            
            if frames_processed % 30 == 0:
                print(f"    Processed {frames_processed} frames...")
    
    cap.release()
    
    print(f"    ✓ Completed: {frames_processed} frames processed, {hand_detected_count} with hand detected")
    
    return {
        "source_video": str(video_p),
        "gesture_name": gesture_name,
        "video_info": {
            "fps": float(fps),
            "total_frames": total_frames,
            "width": width,
            "height": height,
            "duration_sec": total_frames / fps if fps > 0 else 0
        },
        "frames_processed": frames_processed,
        "hand_detected_frames": hand_detected_count,
        "hand_detection_ratio": hand_detected_count / frames_processed if frames_processed > 0 else 0,
        "frames": frame_landmarks
    }

def _iter_video_dirs(base_dir):
    """
    Iterate over directories per letter, each containing videos.
    Expected structure:
    base_dir/
      A/
        video1.mp4
        video2.mp4
      B/
        video1.mp4
      ...
    """
    base_p = Path(base_dir)
    if not base_p.exists():
        raise FileNotFoundError(f"Directory not found: {base_dir}")

    for letter_dir in sorted(base_p.iterdir()):
        if not letter_dir.is_dir():
            continue

        letter = letter_dir.name.strip().upper()
        video_files = []

        # Search for video files
        for ext in ("*.mp4", "*.avi", "*.mov", "*.mkv", "*.flv", "*.webm"):
            video_files.extend(sorted(letter_dir.glob(ext)))

        if video_files:
            yield letter, video_files

def main():
    ap = argparse.ArgumentParser(
        description="Capture MediaPipe landmarks from videos -> single JSON"
    )
    ap.add_argument(
        "--input_dir", 
        default=str(DEFAULT_INPUT_DIR),
        help=f"Root folder with per-letter subfolders containing videos. Default: {DEFAULT_INPUT_DIR}"
    )
    ap.add_argument(
        "--single_output", 
        default=str(BASE_DIR / "mp" / "output_mp" / (Path(__file__).stem + ".json")),
        help="Single JSON file where all video poses will be saved (default: app-back/mp/output_mp)"
    )
    ap.add_argument(
        "--blender", 
        action="store_true", 
        help="Convert world_landmarks to Blender-style coordinates (x,-z,y)"
    )
    ap.add_argument(
        "--sample_frames", 
        type=int, 
        default=1,
        help="Process every Nth frame (e.g. 2 = every 2 frames, speeds up processing)"
    )
    ap.add_argument(
        "--max_frames_per_video", 
        type=int, 
        default=None,
        help="Maximum number of frames to process per video (None = all)"
    )
    args = ap.parse_args()
    
    input_dir = Path(args.input_dir)
    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory not found: {input_dir}")
    
    print("=" * 70)
    print("CAPTURE MediaPipe LANDMARKS FROM VIDEOS")
    print("=" * 70)
    print(f"Input directory: {input_dir}")
    print(f"Blender conversion: {args.blender}")
    print(f"Sample every N frames: {args.sample_frames}")
    print(f"Max frames per video: {args.max_frames_per_video}")
    print()
    
    all_poses = {}
    total_videos = 0
    
    meta = {
        "generated_at": datetime.datetime.utcnow().isoformat() + "Z",
        "source": "MediaPipe hand landmarks from videos",
        "converted_to_blender": bool(args.blender),
        "sample_every_n_frames": args.sample_frames,
        "max_frames_per_video": args.max_frames_per_video
    }
    
    try:
        for letter, video_files in _iter_video_dirs(input_dir):
            print(f"\nProcessing letter: {letter}")
            print(f"  Videos found: {len(video_files)}")
            
            letter_videos = []
            
            for video_file in video_files:
                try:
                    result = extract_hand_landmarks_from_video(
                        str(video_file),
                        gesture_name=letter,
                        convert_to_blender=args.blender,
                        sample_every_n_frames=args.sample_frames,
                        max_frames=args.max_frames_per_video
                    )
                    
                    letter_videos.append(result)
                    total_videos += 1
                    
                except Exception as e:
                    print(f"  ❌ Error processing {video_file.name}: {e}")
            
            if letter_videos:
                # If there is a single video, store directly; if multiple, store as a list
                if len(letter_videos) == 1:
                    all_poses[letter] = letter_videos[0]
                else:
                    all_poses[letter] = letter_videos

                print(f"  ✓ {letter}: {len(letter_videos)} video(s) processed")
    
    except Exception as e:
        print(f"❌ Error during processing: {e}")
        return
    
    if not all_poses:
        print("\n❌ No videos were processed. Check the directory structure.")
        return
    
    # Write single JSON output
    output_file = Path(args.single_output)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    meta["total_videos_processed"] = total_videos
    meta["total_letters"] = len(all_poses)
    
    final = {
        "meta": meta,
        "poses": all_poses
    }
    
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(final, f, indent=2, ensure_ascii=False)
    
    print("\n" + "=" * 70)
    print(f"✅ COMPLETED")
    print(f"   Total videos processed: {total_videos}")
    print(f"   Total letters: {len(all_poses)}")
    print(f"   Output: {output_file}")
    print("=" * 70)

if __name__ == "__main__":
    main()
