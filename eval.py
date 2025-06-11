import torch
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import time
from PIL import Image
import os
import keyboard
from train import CNN
import mss
import numpy as np
import cv2
from collections import deque


IMG_SIZE = 32

transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor()
])
dataset = datasets.ImageFolder(root='./labeled_blocks', transform=transform)

model = CNN()
model.load_state_dict(torch.load('./weights\\spelunky_CNN.pt'))
model.eval()

output_root = "./screenshots/eval"
block_output_root = "./blocks/eval"
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

    print(f"[Split] {count} blocks → {out_dir}")


@logging_time
def separate_inference(images_path):
    print('\n\n separately inferencing...\n')
    labels = []
    for i in range(220):
        image_path = f'./blocks\\eval\\{images_path}\\block_{i}.png'
        img = Image.open(image_path).convert("L").resize((32, 32))
        tensor = transforms.ToTensor()(img).unsqueeze(0)
        pred = model(tensor)
        predicted_class = pred.argmax().item()
        labels.append(predicted_class)
    for row in range(11):
        print(labels[row * 20 : (row + 1) * 20])
    known_map = np.array(labels).reshape(11, 20)
    return known_map


@logging_time
def batch_inference(images_path):
    print('\n\n batch inferencing...\n')
    tensors = []
    for i in range(220):
        dir = f'./blocks\\eval\\{images_path}\\block_{i}.png'
        img = Image.open(dir).convert("L").resize((32, 32))
        tensors.append(transforms.ToTensor()(img))

    batch = torch.stack(tensors)
    with torch.no_grad():
        preds = model(batch)
        pred_classes = preds.argmax(dim=1).tolist()

    labels = pred_classes

    for row in range(11):
        print(labels[row * 20 : (row + 1) * 20])
    known_map = np.array(labels).reshape(11, 20)
    return known_map


@logging_time
def bfs_inference(start_indices, image_folder):
    print('\n\n BFS inferencing...\n')

    HEIGHT, WIDTH = 11, 20
    visited = np.zeros((HEIGHT, WIDTH), dtype=bool)
    known_map = -np.ones((HEIGHT, WIDTH), dtype=int)
    queue = deque([(i // WIDTH, i % WIDTH) for i in start_indices])

    for r, c in queue:
        known_map[r][c] = 0

    def predict_block(r, c):
        path = f'./blocks\\eval\\{image_folder}\\block_{r * WIDTH + c}.png'
        img = Image.open(path).convert("L").resize((32, 32))
        tensor = transforms.ToTensor()(img).unsqueeze(0)
        pred = model(tensor)
        return pred.argmax().item()

    while queue:
        r, c = queue.popleft()
        if not (0 <= r < HEIGHT and 0 <= c < WIDTH):
            continue
        if visited[r][c]:
            continue

        visited[r][c] = True

        if known_map[r][c] == -1:
            known_map[r][c] = predict_block(r, c)

        if known_map[r][c] == 1:
            continue

        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nr, nc = r + dr, c + dc
            if 0 <= nr < HEIGHT and 0 <= nc < WIDTH and not visited[nr][nc]:
                queue.append((nr, nc))

    print("\n[BFS Map]")
    for row in range(HEIGHT):
        print(['.' if known_map[row][col] == 0 else '#' if known_map[row][col] == 1 else '?' for col in range(WIDTH)])
    return known_map


def overlay_known_map_on_image(base_image_path, known_map, block_size=32, save_path=None):

    base = Image.open(base_image_path).convert("RGBA")
    HEIGHT, WIDTH = known_map.shape
    overlay = Image.new("RGBA", base.size, (0, 0, 0, 0))

    for r in range(HEIGHT):
        for c in range(WIDTH):
            x, y = c * block_size, r * block_size
            block_color = (0, 0, 0, 255)

            if known_map[r][c] == 0:
                block_color = (50, 150, 255, 180)
            elif known_map[r][c] == 1:
                block_color = (255, 80, 80, 180)

            block_overlay = Image.new("RGBA", (block_size, block_size), block_color)
            overlay.paste(block_overlay, (x, y))

    result = Image.alpha_composite(base, overlay)
    if save_path:
        result.save(save_path)
        print(f"[Saved Overlay] at {save_path}")
    else:
        result.show()




print(f"Listening for [{key_trigger}] key to eval...")
while True:
    if keyboard.is_pressed(key_trigger):
        try:
            capture_color_frame(output_path=output_root,crop=True, resize_to=(640, 360))
            screenshot_path = capture_frame_and_save(output_path=output_root)
            split_into_blocks(screenshot_path, block_output_root, block_size=block_size)
            results = []
            sep_map = separate_inference(screenshot_path[19:-4])
            batch_map = batch_inference(screenshot_path[19:-4])
            bfs_map = bfs_inference([109, 110], screenshot_path[19:-4])
            overlay_known_map_on_image(screenshot_path, sep_map)
            overlay_known_map_on_image(screenshot_path, batch_map)
            overlay_known_map_on_image(screenshot_path, bfs_map)

        except Exception as e:
            print(f"[Error] {e}")
        time.sleep(1)