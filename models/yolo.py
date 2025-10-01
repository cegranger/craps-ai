import os
from collections import OrderedDict
from queue import Queue

import cv2
import numpy as np
from ultralytics import YOLO


class QueueDict:
    def __init__(self, maxsize):
        self.maxsize = maxsize
        self.data = OrderedDict()

    def put(self, key, value):
        if self.maxsize > 0 and len(self.data) >= self.maxsize:
            raise ValueError("Queue is full.")
        
        self.data[key] = value

    def pop(self):
        if not self.data:
            raise ValueError("Queue is empty.")

        return self.data.popitem(last=False)

    def remove(self, key):
        if key in self.data:
            del self.data[key]

    def keys(self):
        return self.data.keys()

    def values(self):
        return self.data.values()

    def items(self):
        return self.data.items()

    def full(self):
        return self.maxsize > 0 and len(self.data) >= self.maxsize

    def empty(self):
        return not self.data

    def __contains__(self, key):
        return key in self.data

    def __getitem__(self, key):
        return self.data[key]

    def __setitem__(self, key, value):
        self.put(key, value)

    def __delitem__(self, key):
        self.remove(key)

    def __len__(self):
        return len(self.data)

    def __iter__(self):
        return iter(self.data)

    def __repr__(self):
        return repr(self.data)


def train_model(yaml_path, epochs=10, batch_size=16, img_size=640, device="0", project="craps-ai", name="yolov8n"):
    model = YOLO(f"{name}.pt")
    model.train(
        data=yaml_path,
        epochs=epochs,
        batch=batch_size,
        imgsz=img_size,
        device=device,
        project=project,
        name=name,
        # degrees=0.25,
        # scale=0.3,
        # perspective=0.0001
    )

    return model


def load_model(model_path):
    model = YOLO(model_path)
    return model


def take_picture(frame, model, conf=0.5):
    values = []
    crops = []
    results = model.predict(frame, conf=0.5)
    if results:
        result = results[0]
        for box in result.boxes:
            x, y, w, h = box.xywh[0]
            cls_idx = box.cls.item()
            conf = box.conf.item()

            x1 = int(x - w / 2)
            y1 = int(y - h / 2)
            x2 = int(x + w / 2)
            y2 = int(y + h / 2)

            values.append(int(result.names[cls_idx]))
            crops.append(frame[y1:y2, x1:x2, ...])

    return values, crops


def is_within_margin(box1, box2, margin, use_obb):
    if use_obb:
        box1_pts = np.array(box1, dtype=np.float32).reshape(4, 2)
        box2_pts = np.array(box2, dtype=np.float32).reshape(4, 2)

        centroid1 = np.mean(box1_pts, axis=0)
        centroid2 = np.mean(box2_pts, axis=0)

        center_distance = np.linalg.norm(centroid1 - centroid2)

        avg_box_size = (np.linalg.norm(box2_pts[0] - box2_pts[1]) + np.linalg.norm(box2_pts[1] - box2_pts[2])) / 2
        margin_distance = avg_box_size * margin

        if center_distance > margin_distance:
            return False

        for pt in box1_pts:
            distances = np.linalg.norm(box2_pts - pt, axis=1)
            if np.min(distances) > margin_distance:
                return False

        return True
    else:
        x_min1, y_min1, x_max1, y_max1 = box1
        x_min2, y_min2, x_max2, y_max2 = box2

        x_margin = abs(x_max1 - x_min1) * margin
        y_margin = abs(y_max1 - y_min1) * margin

        return (
            abs(x_min1 - x_min2) <= x_margin and
            abs(y_min1 - y_min2) <= y_margin and
            abs(x_max1 - x_max2) <= x_margin and
            abs(y_max1 - y_max2) <= y_margin
        )


def crop_obb_corners(frame, corners):
    corners = np.array(corners, dtype=np.float32)

    width = int(np.linalg.norm(corners[0] - corners[1]))  # Distance between top-left & top-right
    height = int(np.linalg.norm(corners[0] - corners[3]))  # Distance between top-left & bottom-left

    dst_pts = np.array([
        [0, 0],  # Top-left
        [width - 1, 0],  # Top-right
        [width - 1, height - 1],  # Bottom-right
        [0, height - 1]  # Bottom-left
    ], dtype=np.float32)

    M = cv2.getPerspectiveTransform(corners, dst_pts)
    cropped = cv2.warpPerspective(frame, M, (width, height))
    return cropped


def obb_corners_to_xywhr(corners):
    if corners is None or len(corners) == 0:
        raise ValueError("Error: Corners array is empty or None")

    corners = np.array(corners, dtype=np.float32).reshape(-1, 2)

    if corners.shape != (4, 2):
        raise ValueError(f"Error: Expected shape (4,2), but got {corners.shape}")

    rect = cv2.minAreaRect(corners)
    (x_center, y_center), (width, height), angle = rect
    
    # # Ensure angle consistency
    # if width < height:
    #     width, height = height, width
    #     angle += 90

    return np.asarray([x_center, y_center, width, height, angle], dtype=np.float32)


def compute_avg_angles(angles):
    # Compute stable circular mean of angles to prevent 90 degree flips
    angles = np.deg2rad(angles)
    avg_sin = np.mean(np.sin(angles))
    avg_cos = np.mean(np.cos(angles))
    avg_angle = np.arctan2(avg_sin, avg_cos)
    return np.rad2deg(avg_angle)


def weighted_moving_average(values, alpha, recent_first=False):
    # Compute weighted moving average of values, where alpha is the decay factor
    # If recent_first is True, then the first values are weighted more heavily
    if values is None or len(values) == 0:
        raise ValueError("Error: Values array is empty or None")

    values = np.asarray(values)
    if not recent_first:
        values = np.flip(values, axis=0)

    weights = np.power(alpha, np.arange(len(values)))
    weights /= np.sum(weights)
    
    return np.sum(values * weights[:, np.newaxis], axis=0)


def stable_predict(
    result_queue,
    cap,
    model,
    confidence,
    stability_frames,
    position_frames,
    translate_margin,
    debug
):
    if position_frames > stability_frames:
        print(
            """WARNING: Position frames cannot be greater than stability frames.
            Setting position frames to stability frames."""
        )
        position_frames = stability_frames

    previous_detections, current_detections = result_queue

    bbox_tracker = {}
    while cap is not None:
        frame, count = cap.read()
        if not frame:
            # print("Failed to read frame.")
            # break
            continue

        # print(f"Processing frame of shape: {frame.shape}")
        frame_boxes = []

        use_obb = False
        results = model.predict(frame, conf=confidence, verbose=False)
        if results:
            result = results[0]
            if result.obb:
                boxes = result.obb
                use_obb = True
            else:
                boxes = result.boxes

            if not boxes:
                # print("No boxes found.")

                if debug:
                    cv2.imshow("frame", frame)
                    if cv2.waitKey(1) == ord("q"):
                        break

                continue

            for box in boxes:
                if use_obb:
                    p1, p2, p3, p4 = box.xyxyxyxy[0]
                    x1, y1 = int(p1[0]), int(p1[1])
                    x2, y2 = int(p2[0]), int(p2[1])
                    x3, y3 = int(p3[0]), int(p3[1])
                    x4, y4 = int(p4[0]), int(p4[1])

                    position = (x1, y1, x2, y2, x3, y3, x4, y4)

                    # position as key is good enough?
                    # a new dice would have to match the initial position of a previous dice to cause key collision
                    bbox_key = position
                else:
                    x, y, w, h = box.xywh[0]
                    x1, y1 = int(x - w / 2), int(y - h / 2)
                    x2, y2 = int(x + w / 2), int(y + h / 2)

                    position = (x1, y1, x2, y2)
                    bbox_key = position

                frame_boxes.append(bbox_key)
                cls_idx = box.cls.item()
                box_conf = box.conf.item()

                matched = False
                for tracked_box in bbox_tracker:
                    last_position = bbox_tracker[tracked_box]["positions"].queue[-1]
                    if is_within_margin(last_position, bbox_key, margin=translate_margin, use_obb=use_obb):
                        count = bbox_tracker[tracked_box]["count"]
                        bbox_tracker[tracked_box]["count"] = min(count + 1, stability_frames)

                        positions = bbox_tracker[tracked_box]["positions"]
                        if positions.full():
                            positions.get()
                        positions.put(position)
                        bbox_tracker[tracked_box]["positions"] = positions

                        bbox_tracker[tracked_box]["cls"] = result.names[cls_idx]
                        bbox_tracker[tracked_box]["conf"] = box_conf

                        matched = True
                        break

                if not matched:
                    positions = Queue(maxsize=position_frames)
                    positions.put(position)
                    bbox_tracker[bbox_key] = {
                        "count": 1,
                        "positions": positions,
                        "cls": result.names[cls_idx],
                        "conf": box_conf
                    }
        else:
            print("No results found.")

        for tracked_box in list(bbox_tracker.keys()): # copy keys because we may delete from the dict
            deleted = False
            last_position = bbox_tracker[tracked_box]["positions"].queue[-1]
            if not any(is_within_margin(last_position, box, margin=translate_margin, use_obb=use_obb) for box in frame_boxes):
                bbox_tracker[tracked_box]["count"] -= 1
                if bbox_tracker[tracked_box]["count"] <= 0:
                    del bbox_tracker[tracked_box]
                    deleted = True

                    if tracked_box in previous_detections:
                        del previous_detections[tracked_box]
                    if tracked_box in current_detections:
                        del current_detections[tracked_box]

            if not deleted and bbox_tracker[tracked_box]["count"] >= stability_frames:
                positions = np.array(bbox_tracker[tracked_box]["positions"].queue)
                copy_frame = np.array(frame)

                if use_obb:
                    obb_with_angles = np.array([obb_corners_to_xywhr(np.asarray(pos)) for pos in positions])
                    
                    avg_obb = weighted_moving_average(obb_with_angles[:, :4], alpha=0.8, recent_first=False)
                    avg_angle = compute_avg_angles(obb_with_angles[:, 4])

                    # Convert back to corner points
                    avg_rect = ((avg_obb[0], avg_obb[1]), (avg_obb[2], avg_obb[3]), avg_angle)
                    avg_points = cv2.boxPoints(avg_rect).astype(int)

                    if debug:
                        cv2.polylines(frame, [avg_points], isClosed=True, color=(0, 0, 255), thickness=2)

                        text_x, text_y = avg_points[0]
                        text_y = (text_y - 10) if text_y > 10 else (avg_points[3][1] + 30)
                else:
                    avg_x1, avg_y1, avg_x2, avg_y2 = np.mean(positions, axis=0).astype(int)

                    if debug:
                        cv2.rectangle(frame, (avg_x1, avg_y1), (avg_x2, avg_y2), (0, 0, 255), 2)

                        text_x = avg_x1
                        text_y = (avg_y1 - 10) if avg_y1 > 10 else (avg_y2 + 30)

                if debug:
                    cv2.putText(
                        frame,
                        f"{bbox_tracker[tracked_box]['cls']}, {bbox_tracker[tracked_box]['conf']:.2f}",
                        (text_x, text_y),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.9,
                        (0, 255, 0),
                        2
                    )

                if current_detections.full():
                    current_detections.pop()
                current_detections[tracked_box] = (
                    bbox_tracker[tracked_box]["cls"],
                    crop_obb_corners(copy_frame, avg_points.squeeze()) if use_obb\
                    else copy_frame[avg_y1:avg_y2, avg_x1:avg_x2, ...]
                )

        # print(bbox_tracker)

        if debug:
            cv2.imshow("frame", frame)
            if cv2.waitKey(1) == ord("q"):
                break

    cv2.destroyAllWindows()
    return


if __name__ == "__main__":
    camera_source = 0
    epochs = 10
    batch_size = 32
    img_size = 640
    obb_format = True
    train = False

    dataset_path = os.path.join(os.path.dirname(__file__), "..", "datasets")
    yaml_path = os.path.join(dataset_path, "dice_d6.yaml")
    model_name = "yolov8n-obb" if obb_format else "yolov8n" 

    if train:
        yolo_model = train_model(yaml_path, epochs, batch_size, img_size, device="0", name=model_name)
    else:
        yolo_model = load_model(os.path.join(
            os.path.dirname(__file__),
            "craps-ai",
            "yolov8n-obb",
            # "yolov8n4",
            "weights",
            "best.pt"
        ))

    # metrics = yolo_model.val()
    # print(metrics)

    # stable_predict(
    #     yolo_model,
    #     conf=0.6,
    #     stability_frames=20,
    #     position_frames=10,
    #     translate_margin=0.25,
    #     source=camera_source
    # )

    # cap = cv2.VideoCapture(camera_source)
    # while cap.isOpened():
    #     ret, frame = cap.read()
    #     if not ret:
    #         break

    #     results = yolo_model.predict(frame, conf=0.7, verbose=False)
    #     if results:
    #         result = results[0]

    #         use_obb = False
    #         if result.obb:
    #             use_obb = True
    #             boxes = result.obb
    #         else:
    #             boxes = result.boxes

    #         if not boxes:
    #             continue

    #         for box in boxes:
    #             if use_obb:
    #                 p1, p2, p3, p4 = box.xyxyxyxy[0].cpu().numpy()
    #                 x1, y1 = int(p1[0]), int(p1[1])

    #                 points = np.array([p1, p2, p3, p4], np.int32)
    #                 points = points.reshape((-1, 1, 2))

    #                 cv2.polylines(frame, [points], isClosed=True, color=(0, 255, 0), thickness=2)
    #             else:
    #                 x, y, w, h = box.xywh[0]

    #                 x1 = int(x - w / 2)
    #                 y1 = int(y - h / 2)
    #                 x2 = int(x + w / 2)
    #                 y2 = int(y + h / 2)

    #                 cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

    #             cls_idx = box.cls.item()
    #             conf = box.conf.item()

    #             cv2.putText(frame, f"{result.names[cls_idx]}, {conf:.2f}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

    #     cv2.imshow("frame", frame)
    #     if cv2.waitKey(1) == ord("q"):
    #         break

    # cap.release()
    # cv2.destroyAllWindows()
