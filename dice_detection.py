import cv2
import numpy as np
import base64
import io
import json
import time
import os
from PIL import Image
from sklearn.cluster import DBSCAN
from IPython.display import display, Javascript, HTML
from google.colab.output import eval_js
from base64 import b64decode, b64encode

class DiceDetector:
    def __init__(self):
        self.blob_detector = None
        self.capture_stable_count = 0
        self.last_detection = None
        self.params = None

    def create_blob_detector(self, params):
        """Create blob detector with given parameters"""
        detector_params = cv2.SimpleBlobDetector_Params()

        # Filter by Area
        detector_params.filterByArea = True
        detector_params.minArea = params['min_blob_area']
        detector_params.maxArea = params['max_blob_area']

        # Filter by Circularity
        detector_params.filterByCircularity = True
        detector_params.minCircularity = params['circularity'] / 100.0

        # Filter by Convexity
        detector_params.filterByConvexity = True
        detector_params.minConvexity = params['convexity'] / 100.0

        # Filter by Inertia
        detector_params.filterByInertia = True
        detector_params.minInertiaRatio = params['inertia'] / 100.0

        return cv2.SimpleBlobDetector_create(detector_params)

    def detect_dice_with_edges(self, frame, params):
        """Edge detection method"""
        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)

        # Binary thresholding
        _, binary = cv2.threshold(blurred, params['threshold'], 255, cv2.THRESH_BINARY)

        # Morphological operations
        kernel = np.ones((3, 3), np.uint8)
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=2)
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=1)

        # Find contours
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Create overlay for annotations
        overlay = np.zeros((frame.shape[0], frame.shape[1], 4), dtype=np.uint8)

        dice_detections = []

        for contour in contours:
            if cv2.contourArea(contour) < 500:
                continue

            x, y, w, h = cv2.boundingRect(contour)

            # Crop die region
            cropped_die = frame[y:y+h, x:x+w]
            cropped_gray = cv2.cvtColor(cropped_die, cv2.COLOR_BGR2GRAY)
            _, cropped_binary = cv2.threshold(cropped_gray, params['threshold'], 255, cv2.THRESH_BINARY)

            # Detect blobs
            keypoints = self.blob_detector.detect(cropped_binary)
            num_dots = len(keypoints)

            if 1 <= num_dots <= 6:
                dice_detections.append({
                    'x': x, 'y': y, 'w': w, 'h': h,
                    'dots': num_dots,
                    'frame': cropped_gray
                })

                # Draw rectangle and text on overlay
                cv2.rectangle(overlay, (x, y), (x+w, y+h), (0, 255, 0, 255), 2)
                cv2.putText(overlay, f"Value: {num_dots}", (x, y-10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0, 255), 2)

        return binary, overlay, dice_detections

    def detect_dice_with_dbscan(self, frame, params):
        """DBSCAN clustering method"""
        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)

        # Binary thresholding
        _, binary = cv2.threshold(blurred, params['threshold'], 255, cv2.THRESH_BINARY)

        # Detect all blobs
        keypoints = self.blob_detector.detect(binary)

        # Create overlay
        overlay = np.zeros((frame.shape[0], frame.shape[1], 4), dtype=np.uint8)

        dice_detections = []

        if len(keypoints) > 0:
            # Extract coordinates
            points = np.array([kp.pt for kp in keypoints])

            # Cluster using DBSCAN
            clustering = DBSCAN(eps=params['dbscan_eps'], min_samples=1).fit(points)
            labels = clustering.labels_

            for label in set(labels):
                if label == -1:  # Skip noise
                    continue

                cluster_points = points[labels == label]
                x, y = np.mean(cluster_points, axis=0).astype(int)
                num_dots = len(cluster_points)

                if 1 <= num_dots <= 6:
                    # Calculate bounding box for cluster
                    min_x = int(np.min(cluster_points[:, 0]) - 20)
                    max_x = int(np.max(cluster_points[:, 0]) + 20)
                    min_y = int(np.min(cluster_points[:, 1]) - 20)
                    max_y = int(np.max(cluster_points[:, 1]) + 20)

                    # Ensure bounds are within frame
                    min_x = max(0, min_x)
                    min_y = max(0, min_y)
                    max_x = min(frame.shape[1], max_x)
                    max_y = min(frame.shape[0], max_y)

                    dice_detections.append({
                        'x': min_x, 'y': min_y,
                        'w': max_x - min_x, 'h': max_y - min_y,
                        'dots': num_dots,
                        'frame': gray[min_y:max_y, min_x:max_x]
                    })

                    # Draw on overlay
                    cv2.rectangle(overlay, (min_x, min_y), (max_x, max_y), (255, 0, 0, 255), 2)
                    cv2.putText(overlay, f"Value: {num_dots}", (x-20, y-20),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0, 255), 2)
                    cv2.circle(overlay, (x, y), 5, (255, 0, 0, 255), -1)

        return binary, overlay, dice_detections

    def process_frame(self, frame, params):
        """Process a single frame"""
        # Update blob detector if
        if self.params != params:
            self.blob_detector = self.create_blob_detector(params)
            self.params = params
        
        # Choose detection method
        if params['detection_method'] == 0:
            binary, overlay, detections = self.detect_dice_with_edges(frame, params)
        else:
            binary, overlay, detections = self.detect_dice_with_dbscan(frame, params)

        # Handle capture mode
        if params['capture_mode'] and detections:
            self.handle_capture(detections)

        return binary, overlay, detections

    def handle_capture(self, detections):
        """Handle image capture for dataset creation"""
        # Check if detection is stable
        current_detection = tuple(sorted([d['dots'] for d in detections]))

        if self.last_detection == current_detection:
            self.capture_stable_count += 1

            if self.capture_stable_count == 30:  # Stable for 30 frames
                for detection in detections:
                    # Save image
                    timestamp = time.strftime("%Y%m%d-%H%M%S")
                    num_dots = detection['dots']

                    # Create directory if it doesn't exist
                    save_dir = f"datasets/opencv_dataset/{num_dots}"
                    os.makedirs(save_dir, exist_ok=True)

                    # Resize to 64x64
                    resized = cv2.resize(detection['frame'], (64, 64))

                    # Save image
                    filename = f"{save_dir}/{timestamp}.jpg"
                    cv2.imwrite(filename, resized)
                    print(f"Captured: {filename}")

                self.capture_stable_count += 1  # Prevent multiple captures
        else:
            self.capture_stable_count = 0
            self.last_detection = current_detection

# Helper functions for frame conversion
def js_to_image(js_reply):
    """Convert JavaScript image to OpenCV format"""
    if not js_reply or not js_reply.get('img'):
        return None

    # Decode base64 image
    image_bytes = b64decode(js_reply['img'].split(',')[1])
    jpg_as_np = np.frombuffer(image_bytes, dtype=np.uint8)
    img = cv2.imdecode(jpg_as_np, flags=1)

    return img

def image_to_base64(img, format='png'):
    """Convert OpenCV image to base64 string"""
    if img is None:
        return ""

    # Handle different image types
    if len(img.shape) == 2:  # Grayscale
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    elif img.shape[2] == 4:  # RGBA
        pass
    else:  # BGR
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Convert to PIL Image
    pil_img = Image.fromarray(img.astype('uint8'))

    # Save to bytes
    buffer = io.BytesIO()
    pil_img.save(buffer, format=format.upper())

    # Encode to base64
    img_str = base64.b64encode(buffer.getvalue()).decode()
    return f"data:image/{format};base64,{img_str}"

# Show the html layout for the colab notebook
def create_dice_detection_interface():
    display(HTML(filename="dice_detection.html"))

# Main loop
def run_dice_detection():
    """Main function to run the dice detection system"""

    # Create the interface
    create_dice_detection_interface()

    # Initialize detector
    detector = DiceDetector()

    print("Dice Detection System Ready!")
    print("1. Click 'Start Stream' to begin")
    print("2. Adjust parameters as needed")
    print("3. Toggle 'Capture' to save detected dice images")

    # Processing loop
    while True:
        try:
            # Get frame from JavaScript
            js_reply = eval_js('window.getFrame()')

            if not js_reply:
                break

            # Convert to OpenCV image
            frame = js_to_image(js_reply)

            if frame is None:
                continue

            # Get parameters
            params = js_reply.get('params', {})

            # Process frame
            binary, overlay, detections = detector.process_frame(frame, params)

            # Convert to base64
            binary_b64 = image_to_base64(binary)
            overlay_b64 = image_to_base64(overlay)

            # Update display
            eval_js(f'window.updateProcessedFrame("{binary_b64}", "{overlay_b64}")')

            # Update status bar with detection info
            if detections:
                dice_values = [d['dots'] for d in detections]
                status_msg = f"Detected dice: {dice_values}"
                eval_js(f'window.updateStatus("{status_msg}", true)')
            else:
                eval_js('window.updateStatus("No dice detected", false)')

        except KeyboardInterrupt:
            print("\nStopping dice detection...")
            break
        except Exception as e:
            print(f"Error: {e}")
            continue

    print("Dice detection stopped.")