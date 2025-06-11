import mss
import numpy as np
import cv2
import os
import time
from PIL import Image
import keyboard

output_root = "./screenshots"
block_output_root = "./blocks"
block_size = 32
key_trigger = 'F9'

def logging_time(original_fn):
    def wrapper_fn(*args, **kwargs):
        start = time.time()
        result = original_fn(*args, **kwargs)
        end = time.time()
        print(f"[{original_fn.__name__}] took {end - start:.2f} sec")
        return result
    return wrapper_fn

def capture_color_frame(output_path, crop=True, resize_to=(640, 360)):
    with mss.mss() as sct:
        monitor = sct.monitors[1]
        screenshot = sct.grab(monitor)

        img = Image.frombytes('RGB', (screenshot.width, screenshot.height), screenshot.rgb)

        if crop:
            img = img.crop((0, 60, 640, 420))

        if resize_to:
            img = img.resize(resize_to, Image.NEAREST)

        timestamp = int(time.time())
        filename = f"{output_path}/screenshot_{timestamp}.png"
        img.save(filename)
        print(f"Saved screenshot to {filename}")

@logging_time
def capture_frame_and_save(output_path, resize_to=(640, 360)):
    with mss.mss() as sct:
        monitor = sct.monitors[1]
        raw_img = sct.grab(monitor)
        img = np.array(raw_img)[:, :, :3]

        img = img[60:420, 0:640]

        if resize_to:
            img = cv2.resize(img, resize_to, interpolation=cv2.INTER_NEAREST)

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        timestamp = int(time.time())
        filename = f"screenshot_{timestamp}.png"
        path = os.path.join(output_path, filename)

        os.makedirs(output_path, exist_ok=True)
        cv2.imwrite(path, gray)
        print(f"[Saved] {path}")
        return path

@logging_time
def split_into_blocks(image_path, output_dir, block_size=32):
    image = Image.open(image_path)
    width, height = image.size

    base_name = os.path.splitext(os.path.basename(image_path))[0]
    out_dir = os.path.join(output_dir, base_name)
    os.makedirs(out_dir, exist_ok=True)

    count = 0
    for y in range(0, height, block_size):
        for x in range(0, width, block_size):
            block = image.crop((x, y, x + block_size, y + block_size))
            block.save(f"{out_dir}/block_{count}.png")
            count += 1

    print(f"[Split] {count} blocks at {out_dir}")
