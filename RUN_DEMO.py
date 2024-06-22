import os
import re
from PIL import Image, ImageDraw, ImageFont

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

img_folder = "example_data"
out_folder = "eye_contact_results"
# Loop through images in the img_folder and process them
for img_name in os.listdir(img_folder):
    img_path = os.path.join(img_folder, img_name)
    if os.path.isfile(img_path) and img_path.lower().endswith(('.png', '.jpg', '.jpeg')):
        run_on_image(img_path, f"{out_folder}/{img_name}")

print("Processing complete. Check the demo_out folder for results.")
