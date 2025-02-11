import os
from glob import glob

import cv2
import numpy as np

if __name__ == "__main__":
    obb_format = True

    dataset_path = os.path.join(os.path.dirname(__file__), "..", "datasets")
    train_path = os.path.join(dataset_path, "synth-dice-2000-ORIENTED_2025-02-11-13-37-04")

    rng = np.random.default_rng()
    image_paths = glob(os.path.join(train_path, "images", "*.jpg"))
    rng.shuffle(image_paths)

    for i in range(5):
        filename = os.path.basename(image_paths[i])
        label_path = os.path.join(train_path, "labels", filename.replace(".jpg", ".txt"))

        img = cv2.imread(image_paths[i])
        img_height, img_width, img_channels = img.shape
        with open(label_path, "r") as f:
            lines = f.readlines()

        if obb_format:
            for line in lines:
                class_id, x1, y1, x2, y2, x3, y3, x4, y4 = map(float, line.strip().split())

                # Convert to pixel coordinates
                x1, y1 = int(x1 * img_width), int(y1 * img_height)
                x2, y2 = int(x2 * img_width), int(y2 * img_height)
                x3, y3 = int(x3 * img_width), int(y3 * img_height)
                x4, y4 = int(x4 * img_width), int(y4 * img_height)

                # Draw rotated bounding box
                points = np.array([[x1, y1], [x2, y2], [x3, y3], [x4, y4]], np.int32)
                points = points.reshape((-1, 1, 2))
                cv2.polylines(img, [points], isClosed=True, color=(0, 255, 0), thickness=2)
        else:
            for line in lines:
                class_id, x, y, w, h = map(float, line.strip().split())
                x1 = int((x - w / 2) * img_width)
                y1 = int((y - h / 2) * img_height)
                x2 = int((x + w / 2) * img_width)
                y2 = int((y + h / 2) * img_height)

                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)

        cv2.imshow("image", img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
