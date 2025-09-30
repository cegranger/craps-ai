from keras.utils import image_dataset_from_directory
import matplotlib.pyplot as plt
import numpy as np

class Dataset:
    def __init__(self):
        self.dataset_dir = "datasets/opencv_dataset"

        # Image size and parameters
        self.input_shape = (64, 64, 3)
        img_size = (64, 64)
        batch_size = 32
        
        # load dataset from directory
        self.train_data, self.val_data = image_dataset_from_directory(
            directory=self.dataset_dir,
            labels="inferred",
            label_mode="categorical",
            class_names=None,
            color_mode="rgb",
            batch_size=batch_size,
            image_size=img_size,
            shuffle=True,
            seed=42,
            validation_split=0.2,
            subset="both",
            interpolation="bilinear",
            follow_links=False,
            crop_to_aspect_ratio=False,
            pad_to_aspect_ratio=False,
            data_format=None,
            verbose=True,
        )
        
    def plot_classes(self, train_or_val:bool):
        data = self.train_data if train_or_val else self.val_data

        class_names = data.class_names
        num_classes = len(class_names)
        
        images_per_class = {}
        
        for batch_images, batch_labels in data:
            for image, label in zip(batch_images, batch_labels):
                # Convert one-hot encoded label to integer
                label = int(np.argmax(label.numpy()))  # USE np.argmax HERE
                if label not in images_per_class:
                    images_per_class[label] = image
                if len(images_per_class) == num_classes: 
                    break
            if len(images_per_class) == num_classes:
                break
        
        # Plot the images with class names
        fig, axes = plt.subplots(1, num_classes, figsize=(15, 3))

        for i, (label, image) in enumerate(sorted(images_per_class.items())):
            ax = axes[i]
            ax.imshow(image.numpy().astype("uint8")) 
            ax.axis('off')
            ax.set_title(class_names[label])
        
        fig.suptitle("Échantillons des données d'entrainements", fontsize=16, y=0.9) if train_or_val else fig.suptitle("Échantillons des données de validation", fontsize=16, y=0.9)
        plt.tight_layout(rect=[0, 0.05, 1, 0.9])
        plt.show()

