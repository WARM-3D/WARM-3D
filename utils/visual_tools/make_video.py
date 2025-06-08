import glob
import os
from moviepy.editor import ImageSequenceClip


def create_video_from_images(image_folder, output_video_path, fps=12, width=1080):
    """
    Creates a video from a sequence of images.

    Parameters:
    - image_folder: The path to the folder containing the image files.
    - output_video_path: The path where the output video file will be saved.
    - fps: Frames per second for the output video. Default is 12.
    - width: The width of the video. The height will be adjusted automatically. Default is 720.
    """
    # Find all JPEG images in the folder and sort them
    image_files = sorted(glob.glob(os.path.join(image_folder, '*.jpg')))

    # Create a video clip from the images
    clip = ImageSequenceClip(image_files, fps=fps)

    # Resize the clip to the specified width
    clip_resized = clip.resize(width=width)

    # Write the video file to the specified path
    clip_resized.write_videofile(output_video_path, fps=fps)
    # clip_resized.write_videofile(output_video_path, fps=fps, bitrate="3M")


if __name__ == '__main__':
    # Example usage
    image_folder = '/data/output/vis_pred_ema'
    output_video = os.path.join(image_folder, 'video.mp4')
    create_video_from_images(image_folder, output_video)
