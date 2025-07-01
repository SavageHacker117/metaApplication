
import glob
from PIL import Image

# Adjust these for your image grid outputs
image_folder = 'output_*'
out_gif = 'agent_progress.gif'
frame_ms = 500  # 0.5s per frame

images = []
for folder in sorted(glob.glob(image_folder)):
    for img in sorted(glob.glob(f'{folder}/render_*.png')):
        images.append(Image.open(img).convert("RGB"))

if images:
    images[0].save(out_gif, save_all=True, append_images=images[1:], duration=frame_ms, loop=0)
    print(f"Saved GIF: {out_gif}")
else:
    print("No images found!")


