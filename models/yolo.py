import os
from queue import Queue

import cv2
import numpy as np
from ultralytics import YOLO


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


def take_picture(cap, model, conf=0.5):
    ret, frame = cap.read()

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


def stable_predict(model, conf=0.5, stability_frames=5, position_frames=5, margin=0.5, source=0):
    if position_frames > stability_frames:
        print(
            """WARNING: Position frames cannot be greater than stability frames.
            Setting position frames to stability frames."""
        )
        position_frames = stability_frames

    bbox_tracker = {}
    cap = cv2.VideoCapture(source)
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("Failed to read frame.")
            break

        print(f"Processing frame of shape: {frame.shape}")
        frame_boxes = []

        use_obb = False
        results = model.predict(frame, conf=conf, verbose=False)
        if results:
            result = results[0]

            if result.obb:
                boxes = result.obb
                use_obb = True
            else:
                boxes = result.boxes

            if not boxes:
                print("No boxes found.")
                continue

            for box in boxes:
                if use_obb:
                    p1, p2, p3, p4 = box.xyxyxyxy[0]
                    x1, y1 = int(p1[0]), int(p1[1])
                    x2, y2 = int(p2[0]), int(p2[1])
                    x3, y3 = int(p3[0]), int(p3[1])
                    x4, y4 = int(p4[0]), int(p4[1])
                    
                    position = (x1, y1, x2, y2, x3, y3, x4, y4)
                    bbox_key = position # position as key is good enough? a new thrown dice would have to match the exact initial position of a previous dice to cause key collision
                else:
                    x, y, w, h = box.xywh[0]
                    x1, y1 = int(x - w / 2), int(y - h / 2)
                    x2, y2 = int(x + w / 2), int(y + h / 2)

                    position = (x1, y1, x2, y2)
                    bbox_key = position

                frame_boxes.append(bbox_key)
                cls_idx = box.cls.item()
                pred_conf = box.conf.item()

                matched = False
                for tracked_box in bbox_tracker:
                    if is_within_margin(tracked_box, bbox_key, margin=margin, use_obb=use_obb):
                        count = bbox_tracker[tracked_box]["count"]
                        bbox_tracker[tracked_box]["count"] = min(count + 1, stability_frames)

                        positions = bbox_tracker[tracked_box]["positions"]
                        if positions.full():
                            positions.get()
                        positions.put(position)
                        bbox_tracker[tracked_box]["positions"] = positions

                        bbox_tracker[tracked_box]["cls"] = result.names[cls_idx]
                        bbox_tracker[tracked_box]["conf"] = pred_conf

                        matched = True
                        break

                if not matched:
                    positions = Queue(maxsize=position_frames)
                    positions.put(position)
                    bbox_tracker[bbox_key] = {
                        "count": 1,
                        "positions": positions,
                        "cls": result.names[cls_idx],
                        "conf": pred_conf
                    }
        else:
            print("No results found.")

        for tracked_box in list(bbox_tracker.keys()): # copy keys because we may delete from the dict
            deleted = False
            if not any(is_within_margin(tracked_box, box, margin=margin, use_obb=use_obb) for box in frame_boxes):
                bbox_tracker[tracked_box]["count"] -= 1
                if bbox_tracker[tracked_box]["count"] <= 0:
                    del bbox_tracker[tracked_box]
                    deleted = True

            if not deleted and bbox_tracker[tracked_box]["count"] >= stability_frames:
                positions = np.array(bbox_tracker[tracked_box]["positions"].queue)

                if use_obb:
                    avg_points = np.mean(positions.reshape(-1, 4, 2), axis=0).astype(int)
                    avg_points = avg_points.reshape((-1, 1, 2))
                    cv2.polylines(frame, [avg_points], isClosed=True, color=(0, 0, 255), thickness=2)

                    text_x, text_y = avg_points[0, 0]
                    text_y = (text_y - 10) if text_y > 10 else (avg_points[3, 0, 1] + 30)
                else:
                    avg_x1, avg_y1, avg_x2, avg_y2 = np.mean(positions, axis=0).astype(int)
                    cv2.rectangle(frame, (avg_x1, avg_y1), (avg_x2, avg_y2), (0, 0, 255), 2)

                    text_x = avg_x1
                    text_y = (avg_y1 - 10) if avg_y1 > 10 else (avg_y2 + 30)

                cv2.putText(
                    frame,
                    f"{bbox_tracker[tracked_box]['cls']}, {bbox_tracker[tracked_box]['conf']:.2f}",
                    (text_x, text_y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.9,
                    (0, 0, 255),
                    2
                )

        print(bbox_tracker)

        cv2.imshow("frame", frame)
        if cv2.waitKey(1) == ord("q"):
            break

    cap.release()
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
    
    stable_predict(
        yolo_model,
        conf=0.6,
        stability_frames=20,
        position_frames=10,
        margin=0.25,
        source=camera_source
    )

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
    #             if obb_format:
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
