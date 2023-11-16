import os
import cv2
from pathlib import Path
from IPython.display import display, Image as IPImage

input_folder = '/content/images/' # @param {type:"string"}
annotation_folder = '/content/annotation/' # @param {type:"string"}

output_folder = '/content/output/' # @param {type:"string"}
os.makedirs(output_folder, exist_ok=True)

labels = ["face", "hair", "eye", "hands", "feet", "upper_body", "lower_body"]

for img_path in Path(input_folder).glob('*.*'):
    img_name = img_path.stem
    txt_path = os.path.join(annotation_folder, f"{img_name}.txt")

    # Skip if there is no annotation file for the image
    if not os.path.exists(txt_path):
        continue

    img = cv2.imread(str(img_path))

    with open(txt_path, 'r') as txt_file:
        for line in txt_file:
            label_idx, x_center, y_center, width, height = map(float, line.split())
            img_width, img_height, _ = img.shape

            x_center *= img_width
            y_center *= img_height
            width *= img_width
            height *= img_height

            x1 = int(x_center - width / 2)
            y1 = int(y_center - height / 2)
            x2 = int(x_center + width / 2)
            y2 = int(y_center + height / 2)

            label_text = labels[int(label_idx)]

            # Draw bounding box on the image
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)

            # Add label text
            cv2.putText(img, label_text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Convert BGR image to RGB
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Save the image with bounding boxes and labels
    output_path = os.path.join(output_folder, f'{img_name}_annotated.jpg')
    cv2.imwrite(output_path, cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR))

    # Display the image with bounding boxes and labels
    display(IPImage(output_path))
