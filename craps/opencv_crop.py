import json
import cv2
import numpy as np
from sklearn.cluster import DBSCAN

# full_path = 'datasets/craps-ai/'
full_path = ''

# Load parameters from the JSON file
with open(full_path + "datasets/opencv_params.json", "r") as json_file:
    params = json.load(json_file)

# Method parameters
detection_method = params["detection_method"]

# Edge detection parameters
threshold_value = params["edge_detection"]["threshold_value"]  # Range from 0 to 255 in pixel intensity

# Blob detection parameters
min_blob_area = params["blob_detection"]["min_blob_area"]  # Range from 1 pixel to number of pixels in image
max_blob_area = params["blob_detection"]["max_blob_area"]  # Range from 1 pixel to number of pixels in image
min_circularity = params["blob_detection"]["min_circularity"] # Range from 0 to 1, so % is okay
min_convexity =  params["blob_detection"]["min_convexity"]  # Range from 0 to 1, so % is okay
min_inertia = params["blob_detection"]["min_inertia"]  # Range from 0 to 1, so % is okay

# DBSCAN parameters
dbscan_eps = params["dbscan"]["eps"]  # Range from 1 pixel to number of pixels in image

# Set up SimpleBlobDetector parameters

params = cv2.SimpleBlobDetector_Params()
if detection_method == 1:
    # Filter by Area
    params.filterByArea = True
    params.minArea = min_blob_area
    params.maxArea = max_blob_area

    # Filter by Circularity
    params.filterByCircularity = True
    params.minCircularity = min_circularity 

    # Filter by Convexity
    params.filterByConvexity = True
    params.minConvexity = min_convexity 

    # Filter by Inertia
    params.filterByInertia = True
    params.minInertiaRatio = min_inertia 

blob_detector = cv2.SimpleBlobDetector_create(params)


def crop_with_edges(frame):
    # Convert to grayscale and blur
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Binary thresholding
    _, binary = cv2.threshold(blurred, threshold_value, 255, cv2.THRESH_BINARY)

    # Clean up noise
    kernel = np.ones((3, 3), np.uint8)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=2)
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=1)

    # Find contours
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Crop each detected die
    cropped_dice = []
    for i,contour in enumerate(contours):
        if cv2.contourArea(contour) < 500:  # Filter small contours
            continue

        x, y, w, h = cv2.boundingRect(contour)

        # Crop the dice from the original frame
        cropped = frame[y:y + h, x:x + w]
        # cropped = cv2.cvtColor(cropped_die, cv2.COLOR_BGR2GRAY)
        cropped_dice.append(cv2.resize(cropped, (64,64)))

    return cropped_dice


def crop_with_blob(frame):
    # Convert to grayscale and blur
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Binary thresholding
    _, binary = cv2.threshold(blurred, threshold_value, 255, cv2.THRESH_BINARY)

    # Detect blobs (dots) in the binary image
    keypoints = blob_detector.detect(binary)

    # Extract blob coordinates
    points = np.array([kp.pt for kp in keypoints])
    
    cropped_dice = []
    # Cluster blobs using DBSCAN
    if len(points) > 0:
        clustering = DBSCAN(eps=dbscan_eps, min_samples=1).fit(points)
        labels = clustering.labels_

        # Count dice and their dots
        for label in set(labels):
            if label == -1:  # Noise
                continue

            cluster_points = points[labels == label]
            x, y = np.mean(cluster_points, axis=0).astype(int)

            # Cropping the image to 64x64
            x1 = max(0, x - 64 // 2)  # Prevent going out of bounds
            y1 = max(0, y - 64 // 2)
            x2 = min(frame.shape[1], x + 64 // 2)  # Prevent going out of bounds
            y2 = min(frame.shape[0], y + 64 // 2)
            # Convert to grayscale and resize to model's input size
            cropped = frame[y1:y2, x1:x2]
            print('1',cropped.shape)
            # cropped = cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY)  # Grayscale
            cropped = cv2.resize(cropped, (64, 64))
            print('2',cropped.shape)
            cropped_dice.append(cropped)

    return cropped_dice


def take_picture(frame):
    if detection_method == 0:
        cropped_dice = crop_with_edges(frame)
    else:
        cropped_dice = crop_with_blob(frame)
    return cropped_dice

