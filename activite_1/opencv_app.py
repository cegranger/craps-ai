import sys
import cv2
import os
import json
import time
import numpy as np
from sklearn.cluster import DBSCAN
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QImage, QPixmap, QFont
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QLabel, QVBoxLayout, QHBoxLayout, QWidget,
    QSlider, QGroupBox, QDialog, QGridLayout, QSpinBox
)
from qtwidgets import Toggle, AnimatedToggle

full_path = 'k:/activite_1/crap-ai/'
# full_path = ''

class DiceDetectionApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Dice Detection App")
        self.setGeometry(100, 100, 1400, 900)

        # OpenCV Video Capture
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            print("Error: Cannot access webcam.")
            sys.exit()

        # Default parameters
        self.detection_method = 0  # Default to "Method 1: Edge Detection"

        # Edge detection parameters
        self.threshold_value = 50  # Range from 0 to 255 in pixel intensity

        # Blob detection parameters
        self.min_blob_area = 10  # Range from 1 pixel to number of pixels in image
        self.max_blob_area = 50  # Range from 1 pixel to number of pixels in image
        self.min_circularity = 1 # Range from 0 to 1, so % is okay
        self.min_convexity = 1  # Range from 0 to 1, so % is okay
        self.min_inertia = 1  # Range from 0 to 1, so % is okay
        self.dbscan_eps = 50  # Range from 1 pixel to number of pixels in image

        # Blob detector creation
        self.blob_detector = self.create_blob_detector()

        # Capture variables
        self.capture_stable_count = 0
        self.labels = None # array of labels
        self.contours = None # array of contours
        self.dice = None

        # UI Elements
        self.init_ui()

        # Update the blob detector with the right values
        self.update_min_circularity(self.min_circularity)
        self.update_min_convexity(self.min_convexity)
        self.update_min_inertia(self.min_inertia)

        # Timer to update video frames
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frames)
        self.timer.start(30)

    def init_ui(self):
        # Main Layout
        main_layout = QVBoxLayout()

        # Video Streams Layout
        video_layout = QHBoxLayout()

        # Dice Detection Stream
        self.dice_label = QLabel()
        self.dice_label.setFixedSize(640, 480)
        self.dice_label.setStyleSheet("border: 3px solid #3cba54; border-radius: 10px;")
        video_layout.addWidget(self.dice_label)

        # Binary Stream
        self.binary_label = QLabel()
        self.binary_label.setFixedSize(640, 480)
        self.binary_label.setStyleSheet("border: 3px solid #f4c20d; border-radius: 10px;")
        video_layout.addWidget(self.binary_label)

        main_layout.addLayout(video_layout)

        # Parameter Controls with sliders
        param_group = QGroupBox("Paramètres")
        param_group.setStyleSheet("font: bold 14px; color: #3c4043;")
        param_layout = QGridLayout()
        
        # Add toggle switch for capture mode
        self.capture = 0
        self.capture_label = self.add_toggle(param_layout, "Capture", 0, state_label='Off', toggle_function=self.toggle_capture_mode)

        # Add toggle switch for detection method
        self.detection_method = 0
        self.detection_method_label = self.add_toggle(param_layout, "Methode de detection", 1, state_label='par contours', toggle_function=self.toggle_detection_method)

        # Add sliders for each parameter
        self.add_slider(param_layout, "Seuil noir/blanc", 1, 255, self.threshold_value, self.update_threshold, 2)
        self.add_slider(param_layout, "Zone minimale d'un point", 1, 1000, self.min_blob_area, self.update_min_blob_area, 3)
        self.add_slider(param_layout, "Zone maximale d'un point", 1, 1000, self.max_blob_area, self.update_max_blob_area, 4)
        self.add_slider(param_layout, "Circularite", 1, 100, self.min_circularity, self.update_min_circularity, 5)
        self.add_slider(param_layout, "Convexite", 1, 100, self.min_convexity, self.update_min_convexity, 6)
        self.add_slider(param_layout, "Inertie", 1, 100, self.min_inertia, self.update_min_inertia, 7)
        self.add_slider(param_layout, "Zone maximale de regroupement", 1, 1000, self.dbscan_eps, self.update_dbscan_eps, 8)

        # Strech the columns
        param_layout.setColumnStretch(0, 1)  # Label column
        param_layout.setColumnStretch(1, 3)  # Slider or toggle column
        param_layout.setColumnStretch(2, 1)  # Value display column

        param_group.setLayout(param_layout)
        main_layout.addWidget(param_group)

        # Set Main Widget
        widget = QWidget()
        widget.setLayout(main_layout)
        self.setCentralWidget(widget)

        # Apply a colorful style
        self.setStyleSheet("""
            QMainWindow {
                background-color: #f8f9fa;
            }
            QLabel {
                font: bold 12px;
                color: #3c4043;
            }
            QSlider::groove:horizontal {
                height: 6px;
                background: #e0e0e0;
                border-radius: 3px;
            }
            QSlider::handle:horizontal {
                background: #4285f4;
                width: 15px;
                height: 15px;
                margin: -5px 0;
                border-radius: 7px;
            }
            QSlider::handle:horizontal:hover {
                background: #0b66c3;
            }
        """)

    def add_toggle(self, layout, name, row, state_label, toggle_function):
        """Adds a compact toggle switch for detection method."""
        label = QLabel(name)

        # Use the `AnimatedToggle` widget
        toggle = AnimatedToggle()
        toggle.setChecked(False)     # Default to off
        toggle.setFixedSize(60, 40)  # Make the toggle smaller
        toggle.stateChanged.connect(toggle_function)

        # QLabel to display the current state of the toggle
        state_label = QLabel(state_label)
        state_label.setStyleSheet("font: bold 13px; color: #3c4043;")

        # Add the label and toggle to the layout
        layout.addWidget(label, row, 0, Qt.AlignLeft)
        layout.addWidget(toggle, row, 1, Qt.AlignLeft)
        layout.addWidget(state_label, row, 2, Qt.AlignLeft)

        # Add a spacer to fill the row and prevent stretching
        layout.setColumnStretch(2, 1)

        return state_label


    def add_slider(self, layout, name, min_val, max_val, default_val, callback, row):
        """Adds a labeled slider with a value display."""
        label = QLabel(name)
        slider = QSlider(Qt.Horizontal)
        slider.setMinimum(min_val)
        slider.setMaximum(max_val)
        slider.setValue(default_val)
        slider.valueChanged.connect(callback)

        # Value Display using QSpinBox
        value_display = QSpinBox()
        value_display.setRange(min_val, max_val)  # Set the range to match the slider
        value_display.setValue(default_val)  # Initialize with the default value
        value_display.setAlignment(Qt.AlignCenter)  # Center-align the text
        value_display.setStyleSheet("""
            QSpinBox {
                font-size: 18px;  /* Set font size */
                font-weight: bold;  /* Make text bold */
            }
        """)

        slider.valueChanged.connect(value_display.setValue)
        value_display.valueChanged.connect(slider.setValue)

        layout.addWidget(label, row, 0)
        layout.addWidget(slider, row, 1)
        layout.addWidget(value_display, row, 2)

        # Save references to specific sliders for min/max blob area
        if name == "Zone minimale d'un point":
            self.min_blob_slider = slider
        elif name == "Zone maximale d'un point":
            self.max_blob_slider = slider

        return slider
    
    
    def toggle_capture_mode(self, state):
        """Switch between detection methods."""
        if state == Qt.Checked:
            self.capture = 1
            self.capture_label.setText("On")
            self.save_to_json()
        else:
            self.capture = 0
            self.capture_label.setText("Off")


    def toggle_detection_method(self, state):
        """Switch between detection methods."""
        if state == Qt.Checked:
            self.detection_method = 1  # Blob Count + DBSCAN
            self.detection_method_label.setText("par regroupements")
        else:
            self.detection_method = 0  # Edge Detection
            self.detection_method_label.setText("par contours")

    def update_threshold(self, value):
        self.threshold_value = value

    def update_min_blob_area(self, value):
        self.min_blob_area = value

        # Ensure min_blob_area is not greater than max_blob_area
        if self.min_blob_area > self.max_blob_area:
            self.max_blob_area = self.min_blob_area
            # Update the max blob area slider to match the min blob area
            self.max_blob_slider.setValue(self.max_blob_area)

        # Update the blob detector
        self.blob_detector = self.create_blob_detector()

    def update_max_blob_area(self, value):
        self.max_blob_area = value

        # Ensure max_blob_area is not less than min_blob_area
        if self.max_blob_area < self.min_blob_area:
            self.min_blob_area = self.max_blob_area
            # Update the min blob area slider to match the max blob area
            self.min_blob_slider.setValue(self.min_blob_area)

        # Update the blob detector
        self.blob_detector = self.create_blob_detector()

    def update_min_circularity(self, value):
        self.min_circularity = value / 100.0  # Convert to float (0-1)
        self.blob_detector = self.create_blob_detector()

    def update_min_convexity(self, value):
        self.min_convexity = value / 100.0  # Convert to float (0-1)
        self.blob_detector = self.create_blob_detector()

    def update_min_inertia(self, value):
        self.min_inertia = value / 100.0  # Convert to float (0-1)
        self.blob_detector = self.create_blob_detector()

    def update_blob_color(self, value):
        self.blob_color = value
        self.blob_detector = self.create_blob_detector()
    
    def update_dbscan_eps(self, value):
        self.dbscan_eps = value

    def create_blob_detector(self):
        # Set up SimpleBlobDetector parameters
        params = cv2.SimpleBlobDetector_Params()

        # Filter by Area
        params.filterByArea = True
        params.minArea = self.min_blob_area
        params.maxArea = self.max_blob_area

        # Filter by Circularity
        params.filterByCircularity = True
        params.minCircularity = self.min_circularity 

        # Filter by Convexity
        params.filterByConvexity = True
        params.minConvexity = self.min_convexity 

        # Filter by Inertia
        params.filterByInertia = True
        params.minInertiaRatio = self.min_inertia 

        return cv2.SimpleBlobDetector_create(params)

    def detect_dice_with_edges(self, frame):
        # Convert to grayscale and blur
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)

        # Binary thresholding
        _, binary = cv2.threshold(blurred, self.threshold_value, 255, cv2.THRESH_BINARY)

        # Morphological operations to clean up noise
        kernel = np.ones((3, 3), np.uint8)
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=2)
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=1)

        # Find contours
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Process each detected die
        for i,contour in enumerate(contours):
            if cv2.contourArea(contour) < 500:  # Filter small contours
                continue

            x, y, w, h = cv2.boundingRect(contour)

            # Crop the dice from the original frame
            cropped_die = frame[y:y + h, x:x + w]

            cropped_gray = cv2.cvtColor(cropped_die, cv2.COLOR_BGR2GRAY)
            _, cropped_binary = cv2.threshold(cropped_gray, self.threshold_value, 255, cv2.THRESH_BINARY)

            # Detect blobs (dots) in the binary image
            keypoints = self.blob_detector.detect(cropped_binary)
            num_dots = len(keypoints)

            # Capture logic
            if self.capture and num_dots in range(1,7): # num dots over 6 means many dice together.
                if self.capture_stable_count == 40:
                    # Resizing the crop to 64x64
                    cropped_frame = cv2.resize(cropped_gray, (64,64))
                    # Check if the crop is valid
                    if cropped_frame.size == 0:
                        print('CALL AN AMBULANCE! cropped image size is 0.')
                    else:
                        directory = f"activite_1/opencv_dataset/{num_dots}"
                        directory = full_path + directory
                        os.makedirs(directory, exist_ok=True)
                        timestamp = time.strftime("%Y%m%d-%H%M%S")
                        filename = f"{directory}/{timestamp}.jpg"
                        print(cv2.imwrite(filename, cropped_frame))
                        print(f"Image enregistree: {filename}")
                        self.show_popup(f"Image enregistree: {filename}")
                    self.capture_stable_count += 1 # continue counting until new dice throw and reset to 0.
                elif self.capture_stable_count == 0:   #   jumpstart the capture loop
                    self.contours = num_dots
                    self.capture_stable_count += 1
                elif num_dots == self.contours:
                    self.capture_stable_count += 1
                else:
                    self.capture_stable_count = 0
                self.contours = num_dots


            # Draw keypoints (dots) and rectangle
            cv2.putText(frame, f"Value: {num_dots}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # Display the output
        self.display_image(frame, self.dice_label)
        self.display_image(binary, self.binary_label, is_binary=True)

    def detect_dice_with_dbscan(self, frame):
        # Convert to grayscale and blur
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)

        # Binary thresholding
        _, binary = cv2.threshold(blurred, self.threshold_value, 255, cv2.THRESH_BINARY)

        # Detect blobs (dots) in the binary image
        keypoints = self.blob_detector.detect(binary)

        # Extract blob coordinates
        points = np.array([kp.pt for kp in keypoints])  # (x, y) coordinates of blobs

        # Cluster blobs using DBSCAN
        if len(points) > 0:
            clustering = DBSCAN(eps=self.dbscan_eps, min_samples=1).fit(points)
            labels = clustering.labels_

            # Count dice and their dots
            for label in set(labels):
                if label == -1:  # Noise
                    continue

                cluster_points = points[labels == label]
                x, y = np.mean(cluster_points, axis=0).astype(int)
                num_dots = len(cluster_points)

                # Capture logic
                if self.capture and num_dots in range(1,7): # num dots over 6 means many dice together.
                    if self.capture_stable_count == 30:
                        # Cropping the image to 64x64
                        x1 = max(0, x - 64 // 2)  # Prevent going out of bounds
                        y1 = max(0, y - 64 // 2)
                        x2 = min(frame.shape[1], x + 64 // 2)  # Prevent going out of bounds
                        y2 = min(frame.shape[0], y + 64 // 2)
                        cropped_frame = cv2.cvtColor(frame[y1:y2, x1:x2], cv2.COLOR_BGR2GRAY)
                        # Check if the crop is valid
                        if cropped_frame.size == 0:
                            print('CALL AN AMBULANCE! cropped image size is 0.')
                        else:
                            directory = f"activite_1/opencv_dataset/{num_dots}"
                            directory = full_path + directory
                            os.makedirs(directory, exist_ok=True)
                            timestamp = time.strftime("%Y%m%d-%H%M%S")
                            filename = 'test.jpg' #f"{directory}/{timestamp}.jpg"
                            print(cv2.imwrite(filename, cropped_frame))
                            print(f"Image enregistree: {filename}")
                            self.show_popup(f"Image enregistree: {filename}")
                        self.capture_stable_count += 1 # continue counting until new dice throw and reset to 0.
                    elif self.capture_stable_count == 0:   #   jumpstart the capture loop
                        self.labels = labels
                        self.capture_stable_count += 1
                    elif np.array_equal(labels, self.labels):
                        self.capture_stable_count += 1
                    else:
                        self.capture_stable_count = 0
                    self.labels = labels
                
                # Draw the cluster information
                cv2.putText(frame, f"Value: {num_dots}", (x - 20, y - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                cv2.circle(frame, (x, y), 5, (0, 0, 255), -1)
                        

        # Display the output
        self.display_image(frame, self.dice_label)
        self.display_image(binary, self.binary_label, is_binary=True)

    def show_popup(self, message):
        """Show a popup message and close it after 1 second."""
        # Create a popup window (QDialog)
        popup = QDialog(self)
        popup.setWindowFlags(Qt.FramelessWindowHint | Qt.Dialog)
        popup.setAttribute(Qt.WA_TranslucentBackground)
        popup.setModal(False)  # Allow interaction with the main window
        popup.setGeometry(
            self.geometry().x() + self.width() // 2 - 150,
            self.geometry().y() + self.height() // 2 + 75,
            300,
            100,
        )

        # Add a label to the popup
        message_label = QLabel(message, popup)
        message_label.setAlignment(Qt.AlignCenter)
        message_label.setFont(QFont("Courrier", 24, QFont.Bold))
        message_label.setStyleSheet(
            "color: white; background-color: rgba(0, 0, 0, 180); padding: 10px; border-radius: 5px;"
        )

        # Set layout for the popup
        layout = QVBoxLayout()
        layout.addWidget(message_label)
        popup.setLayout(layout)

        # Show the popup
        popup.show()

        # Close the popup after 1 second
        QTimer.singleShot(1000, popup.close)

    def save_to_json(self, file_name="activite_1/opencv_params.json"):
        """Save the class attributes to a JSON file."""
        # Create a dictionary from the class attributes
        data = {
            "detection_method": self.detection_method,
            "edge_detection": {
                "threshold_value": self.threshold_value
            },
            "blob_detection": {
                "min_blob_area": self.min_blob_area,
                "max_blob_area": self.max_blob_area,
                "min_circularity": self.min_circularity,
                "min_convexity": self.min_convexity,
                "min_inertia": self.min_inertia
            },
            "dbscan": {
                "eps": self.dbscan_eps
            }
        }

        # Write the dictionary to a JSON file
        with open(file_name, "w") as json_file:
            json.dump(data, json_file, indent=4)

        print(f"Parameters saved to {file_name}")


    def update_frames(self):
        ret, frame = self.cap.read()
        if not ret:
            return

        # Resize frame for easier processing
        frame = cv2.resize(frame, (800, 600))

        # Use the selected detection method
        if self.detection_method == 0:  # Method 1: Edge Detection, Crop, and Blob Count
            self.detect_dice_with_edges(frame)
        elif self.detection_method == 1:  # Method 2: Blob Count + DBSCAN
            self.detect_dice_with_dbscan(frame)

    def display_image(self, img, label, is_binary=False):
        if is_binary:
            qformat = QImage.Format_Grayscale8
        else:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            qformat = QImage.Format_RGB888

        height, width = img.shape[:2]
        img = QImage(img.data, width, height, img.strides[0], qformat)
        label.setPixmap(QPixmap.fromImage(img).scaled(label.width(), label.height(), Qt.KeepAspectRatio))

    def closeEvent(self, event):
        self.cap.release()
        cv2.destroyAllWindows()
        event.accept()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = DiceDetectionApp()
    window.show()
    sys.exit(app.exec_())