import os
from glob import glob

import cv2
import numpy as np

if __name__ == "__main__":
    dataset_path = os.path.join(os.path.dirname(__file__), "..", "datasets")
    train_path = os.path.join(dataset_path, "dice_d6", "train")

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
