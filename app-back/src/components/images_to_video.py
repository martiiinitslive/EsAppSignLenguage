"""
Module to stitch images into a video.
"""

from moviepy.editor import ImageClip, concatenate_videoclips

def images_to_video(images_list, output_video_path, duration=1):
    clips = [ImageClip(img).set_duration(duration) for img in images_list]
    # Optionally add cross-fade transitions between clips
    final_clip = concatenate_videoclips(clips, method="compose", padding=-duration/2)
    final_clip.write_videofile(output_video_path, fps=24)
