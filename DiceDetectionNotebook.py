import cv2
import numpy as np
import ipywidgets as widgets
from IPython.display import display, Image
import io
from PIL import Image as PILImage
import threading
import time
import json
import os
from sklearn.cluster import DBSCAN

class DiceDetectionNotebook:
    def __init__(self):
        # OpenCV Video Capture
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            print("Error: Cannot access webcam.")
            return
        
        # Default parameters
        self.detection_method = 0  # Default to "Method 1: Edge Detection"
        
        # Edge detection parameters
        self.threshold_value = 50
        
        # Blob detection parameters
        self.min_blob_area = 10
        self.max_blob_area = 50
        self.min_circularity = 0.01
        self.min_convexity = 0.01
        self.min_inertia = 0.01
        self.dbscan_eps = 50
        
        # Capture variables
        self.capture_mode = False
        self.capture_stable_count = 0
        self.labels = None
        self.contours = None
        
        # Create blob detector
        self.blob_detector = self.create_blob_detector()
        
        # Video streaming flag
        self.streaming = False
        
        # Initialize UI
        self.init_ui()
        
    def init_ui(self):
        """Initialize the notebook UI with widgets"""
        
        # Video display widgets
        self.dice_output = widgets.Image(
            format='jpeg',
            width=640,
            height=480,
            layout=widgets.Layout(border='3px solid #3cba54', border_radius='10px')
        )
        
        self.binary_output = widgets.Image(
            format='jpeg',
            width=640,
            height=480,
            layout=widgets.Layout(border='3px solid #f4c20d', border_radius='10px')
        )
        
        # Create control widgets
        
        # Capture toggle
        self.capture_toggle = widgets.ToggleButton(
            value=False,
            description='Capture: Off',
            button_style='',
            tooltip='Toggle capture mode',
            icon='camera'
        )
        self.capture_toggle.observe(self.on_capture_toggle, names='value')
        
        # Detection method toggle
        self.method_toggle = widgets.ToggleButton(
            value=False,
            description='Method: Contours',
            button_style='',
            tooltip='Toggle detection method'
        )
        self.method_toggle.observe(self.on_method_toggle, names='value')
        
        # Threshold slider
        self.threshold_slider = widgets.IntSlider(
            value=self.threshold_value,
            min=1,
            max=255,
            step=1,
            description='Threshold:',
            style={'description_width': 'initial'},
            layout=widgets.Layout(width='500px')
        )
        self.threshold_slider.observe(self.on_threshold_change, names='value')
        
        # Blob area sliders
        self.min_area_slider = widgets.IntSlider(
            value=self.min_blob_area,
            min=1,
            max=1000,
            step=1,
            description='Min Blob Area:',
            style={'description_width': 'initial'},
            layout=widgets.Layout(width='500px')
        )
        self.min_area_slider.observe(self.on_min_area_change, names='value')
        
        self.max_area_slider = widgets.IntSlider(
            value=self.max_blob_area,
            min=1,
            max=1000,
            step=1,
            description='Max Blob Area:',
            style={'description_width': 'initial'},
            layout=widgets.Layout(width='500px')
        )
        self.max_area_slider.observe(self.on_max_area_change, names='value')
        
        # Blob detection parameters
        self.circularity_slider = widgets.IntSlider(
            value=int(self.min_circularity * 100),
            min=1,
            max=100,
            step=1,
            description='Circularity %:',
            style={'description_width': 'initial'},
            layout=widgets.Layout(width='500px')
        )
        self.circularity_slider.observe(self.on_circularity_change, names='value')
        
        self.convexity_slider = widgets.IntSlider(
            value=int(self.min_convexity * 100),
            min=1,
            max=100,
            step=1,
            description='Convexity %:',
            style={'description_width': 'initial'},
            layout=widgets.Layout(width='500px')
        )
        self.convexity_slider.observe(self.on_convexity_change, names='value')
        
        self.inertia_slider = widgets.IntSlider(
            value=int(self.min_inertia * 100),
            min=1,
            max=100,
            step=1,
            description='Inertia %:',
            style={'description_width': 'initial'},
            layout=widgets.Layout(width='500px')
        )
        self.inertia_slider.observe(self.on_inertia_change, names='value')
        
        # DBSCAN epsilon slider
        self.dbscan_slider = widgets.IntSlider(
            value=self.dbscan_eps,
            min=1,
            max=1000,
            step=1,
            description='DBSCAN eps:',
            style={'description_width': 'initial'},
            layout=widgets.Layout(width='500px')
        )
        self.dbscan_slider.observe(self.on_dbscan_change, names='value')
        
        # Start/Stop button
        self.start_button = widgets.Button(
            description='Start Stream',
            button_style='success',
            tooltip='Start video stream',
            icon='play'
        )
        self.start_button.on_click(self.toggle_stream)
        
        # Status label
        self.status_label = widgets.Label(value='Ready')
        
        # Layout the widgets
        video_box = widgets.HBox([self.dice_output, self.binary_output])
        
        controls_box = widgets.VBox([
            widgets.HBox([self.start_button, self.status_label]),
            widgets.HBox([self.capture_toggle, self.method_toggle]),
            self.threshold_slider,
            self.min_area_slider,
            self.max_area_slider,
            self.circularity_slider,
            self.convexity_slider,
            self.inertia_slider,
            self.dbscan_slider
        ])
        
        # Main container
        self.main_container = widgets.VBox([
            widgets.HTML("<h2>Dice Detection App</h2>"),
            video_box,
            widgets.HTML("<h3>Parameters</h3>"),
            controls_box
        ])
        
        display(self.main_container)
    
    def on_capture_toggle(self, change):
        self.capture_mode = change['new']
        if self.capture_mode:
            self.capture_toggle.description = 'Capture: On'
            self.capture_toggle.button_style = 'success'
            self.save_to_json()
        else:
            self.capture_toggle.description = 'Capture: Off'
            self.capture_toggle.button_style = ''
    
    def on_method_toggle(self, change):
        if change['new']:
            self.detection_method = 1
            self.method_toggle.description = 'Method: DBSCAN'
            self.method_toggle.button_style = 'info'
        else:
            self.detection_method = 0
            self.method_toggle.description = 'Method: Contours'
            self.method_toggle.button_style = ''
    
    def on_threshold_change(self, change):
        self.threshold_value = change['new']
    
    def on_min_area_change(self, change):
        self.min_blob_area = change['new']
        if self.min_blob_area > self.max_blob_area:
            self.max_blob_area = self.min_blob_area
            self.max_area_slider.value = self.max_blob_area
        self.blob_detector = self.create_blob_detector()
    
    def on_max_area_change(self, change):
        self.max_blob_area = change['new']
        if self.max_blob_area < self.min_blob_area:
            self.min_blob_area = self.max_blob_area
            self.min_area_slider.value = self.min_blob_area
        self.blob_detector = self.create_blob_detector()
    
    def on_circularity_change(self, change):
        self.min_circularity = change['new'] / 100.0
        self.blob_detector = self.create_blob_detector()
    
    def on_convexity_change(self, change):
        self.min_convexity = change['new'] / 100.0
        self.blob_detector = self.create_blob_detector()
    
    def on_inertia_change(self, change):
        self.min_inertia = change['new'] / 100.0
        self.blob_detector = self.create_blob_detector()
    
    def on_dbscan_change(self, change):
        self.dbscan_eps = change['new']
    
    def create_blob_detector(self):
        params = cv2.SimpleBlobDetector_Params()
        params.filterByArea = True
        params.minArea = self.min_blob_area
        params.maxArea = self.max_blob_area
        params.filterByCircularity = True
        params.minCircularity = self.min_circularity
        params.filterByConvexity = True
        params.minConvexity = self.min_convexity
        params.filterByInertia = True
        params.minInertiaRatio = self.min_inertia
        return cv2.SimpleBlobDetector_create(params)
    
    def toggle_stream(self, button):
        if not self.streaming:
            self.streaming = True
            self.start_button.description = 'Stop Stream'
            self.start_button.button_style = 'danger'
            self.start_button.icon = 'stop'
            self.status_label.value = 'Streaming...'
            # Start streaming in a separate thread
            self.stream_thread = threading.Thread(target=self.stream_video)
            self.stream_thread.daemon = True
            self.stream_thread.start()
        else:
            self.streaming = False
            self.start_button.description = 'Start Stream'
            self.start_button.button_style = 'success'
            self.start_button.icon = 'play'
            self.status_label.value = 'Stopped'
    
    def stream_video(self):
        while self.streaming:
            ret, frame = self.cap.read()
            if not ret:
                continue
            
            # Resize frame
            frame = cv2.resize(frame, (800, 600))
            
            # Process frame based on detection method
            if self.detection_method == 0:
                self.detect_dice_with_edges(frame)
            else:
                self.detect_dice_with_dbscan(frame)
            
            time.sleep(0.03)  # ~30 FPS
    
    def detect_dice_with_edges(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        _, binary = cv2.threshold(blurred, self.threshold_value, 255, cv2.THRESH_BINARY)
        
        kernel = np.ones((3, 3), np.uint8)
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=2)
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=1)
        
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for contour in contours:
            if cv2.contourArea(contour) < 500:
                continue
            
            x, y, w, h = cv2.boundingRect(contour)
            cropped_die = frame[y:y + h, x:x + w]
            cropped_gray = cv2.cvtColor(cropped_die, cv2.COLOR_BGR2GRAY)
            _, cropped_binary = cv2.threshold(cropped_gray, self.threshold_value, 255, cv2.THRESH_BINARY)
            
            keypoints = self.blob_detector.detect(cropped_binary)
            num_dots = len(keypoints)
            
            # Handle capture logic here if needed
            
            cv2.putText(frame, f"Value: {num_dots}", (x, y - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        
        self.update_displays(frame, binary)
    
    def detect_dice_with_dbscan(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        _, binary = cv2.threshold(blurred, self.threshold_value, 255, cv2.THRESH_BINARY)
        
        keypoints = self.blob_detector.detect(binary)
        points = np.array([kp.pt for kp in keypoints])
        
        if len(points) > 0:
            clustering = DBSCAN(eps=self.dbscan_eps, min_samples=1).fit(points)
            labels = clustering.labels_
            
            for label in set(labels):
                if label == -1:
                    continue
                
                cluster_points = points[labels == label]
                x, y = np.mean(cluster_points, axis=0).astype(int)
                num_dots = len(cluster_points)
                
                cv2.putText(frame, f"Value: {num_dots}", (x - 20, y - 20), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                cv2.circle(frame, (x, y), 5, (0, 0, 255), -1)
        
        self.update_displays(frame, binary)
    
    def update_displays(self, frame, binary):
        # Convert frame to JPEG for display
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        _, frame_jpg = cv2.imencode('.jpg', frame_rgb)
        self.dice_output.value = frame_jpg.tobytes()
        
        # Convert binary to JPEG for display
        _, binary_jpg = cv2.imencode('.jpg', binary)
        self.binary_output.value = binary_jpg.tobytes()
    
    def save_to_json(self, file_name="dice_params.json"):
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
        
        with open(file_name, "w") as json_file:
            json.dump(data, json_file, indent=4)
        
        self.status_label.value = f"Parameters saved to {file_name}"
    
    def close(self):
        """Clean up resources"""
        self.streaming = False
        if self.cap.isOpened():
            self.cap.release()
        cv2.destroyAllWindows()

# To use in a Jupyter notebook:
# app = DiceDetectionNotebook()
# app.close() when finished
