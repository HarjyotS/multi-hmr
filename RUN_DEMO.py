import os
import shutil
import re
import subprocess
from PIL import Image, ImageDraw, ImageFont

# CHANGE!!!
path_to_video = "test_video.mp4"
output_video_path = "OUTPUT_VIDEO.mp4"
video_frames_folder = "video_frames"
video_output_frames_folder = "video_output_frames"

def run_on_image(image_path, output_path):
    # Step 1: Run the command to execute demosuper.py
    command = f"python demosuper.py --img_path \"{image_path}\" --out_folder demo_out --extra_views 1 --model_name multiHMR_896_L --distance 1"
    os.system(command)

    # Step 2: Read the verts.txt file to check for eye contact
    verts_file_path = "verts.txt"

    with open(verts_file_path, 'r') as file:
        content = file.read()

    eye_contact_match = re.search(r"EYE CONTACT: (True|False)", content)
    eye_contact = eye_contact_match.group(1) if eye_contact_match else "Unknown"

    # Function to add text to an image
    def add_text_to_image(image_path, text, output_path):
        image = Image.open(image_path)
        draw = ImageDraw.Draw(image)
        font = ImageFont.load_default()
        font_size = int(image.height / 20)
        font = ImageFont.truetype("arial.ttf", font_size)

        # Position for the text (bottom-right corner)
        text_position = (image.width - 150, image.height - 30)
        # Make text never go out of bounds
        text_position = (min(text_position[0], image.width - 150), min(text_position[1], image.height - 30))

        # Add text to image
        draw.text(text_position, text, font=font, fill=(255, 0, 0))

        # Save the modified image
        image.save(output_path)

    add_text_to_image(image_path, f"{eye_contact}", output_path)

def process_video(video_path):
    # Step 1: Extract frames from the video at 10 fps
    os.makedirs(video_frames_folder, exist_ok=True)
    os.makedirs(video_output_frames_folder, exist_ok=True)
    
    subprocess.call([
        "ffmpeg", "-i", video_path, "-vf", "fps=10", f"{video_frames_folder}/frame_%04d.png"
    ], shell=True)

    # Step 2: Process each frame with the eye contact script
    for frame_name in os.listdir(video_frames_folder):
        frame_path = os.path.join(video_frames_folder, frame_name)
        if os.path.isfile(frame_path) and frame_path.lower().endswith('.png'):
            output_frame_path = os.path.join(video_output_frames_folder, frame_name)
            run_on_image(frame_path, output_frame_path)

    # Step 3: Compile the processed frames back into a video
    subprocess.call([
        "ffmpeg", "-framerate", "10", "-i", f"{video_output_frames_folder}/frame_%04d.png", "-c:v", "libx264", "-pix_fmt", "yuv420p", output_video_path
    ], shell=True)

    # Step 4: Clean up temporary folders
    shutil.rmtree(video_frames_folder)
    shutil.rmtree(video_output_frames_folder)

# Example usage
process_video(path_to_video)

print("Video processing complete. Check the output_video.mp4 for results.")