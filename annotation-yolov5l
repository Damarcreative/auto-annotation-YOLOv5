import torch
from pathlib import Path
import os
from PIL import Image
import cv2


custom_weights = '/content/drive/MyDrive/runs/train/exp/weights/best.pt'
input_folder = '/content/input/'
output_folder = '/content/output/'
expand_box = 1
render_box = False
print_log = False



os.makedirs(output_folder, exist_ok=True)

model = torch.hub.load('ultralytics/yolov5', 'custom', path=custom_weights, force_reload=True)

def adjust_bbox(bbox, expand_factor, img_width, img_height):
    x1, y1, x2, y2 = bbox
    width = x2 - x1
    height = y2 - y1
    new_width = width * expand_factor
    new_height = height * expand_factor
    new_x1 = max(x1 - (new_width - width) / 2, 0)
    new_y1 = max(y1 - (new_height - height) / 2, 0)
    new_x2 = min(x2 + (new_width - width) / 2, img_width)
    new_y2 = min(y2 + (new_height - height) / 2, img_height)
    return new_x1, new_y1, new_x2, new_y2

for img_path in Path(input_folder).glob('*.*'):
    # Use cv2.imread to read the image
    original_img = cv2.imread(str(img_path))
    if original_img is None:
        continue  # Skip if the image cannot be read

    results = model(original_img)

    img_name = img_path.stem
    txt_name = img_name + ".txt"

    txt_path = os.path.join(output_folder, txt_name)

    img_height, img_width, _ = original_img.shape

    with open(txt_path, 'w') as txt_file:
        for idx, det in enumerate(results.xyxy[0]):
            bbox = adjust_bbox(det[:4].cpu().numpy(), expand_box, img_width, img_height)

            # x_center, y_center, width, height = [(bbox[i] + bbox[i + 2]) / 2 for i in range(4)]

            x_center, y_center, width, height = [(bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2, bbox[2] - bbox[0], bbox[3] - bbox[1]]
            x_center /= img_width
            y_center /= img_height
            width /= img_width
            height /= img_height

            label_idx = int(det[5])
            label = ["face", "hair", "eye", "hands", "feet", "upper_body", "lower_body"][label_idx]

            line = f"{label_idx} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n"
            txt_file.write(line)

    if print_log:
        print(f'Processed {img_path}')
        #------------------------------------------------------------ok
