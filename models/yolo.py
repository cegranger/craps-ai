import os
from queue import Queue

import cv2
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

def is_within_margin(box1, box2, margin):
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

def stable_predict(model, conf=0.5, stability_frames=5, position_frames=5, margin=0.5):
    if position_frames > stability_frames:
        print(
            """WARNING: Position frames cannot be greater than stability frames.
            Setting position frames to stability frames."""
        )
        position_frames = stability_frames

    bbox_tracker = {}
    cap = cv2.VideoCapture(1)
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("Failed to read frame.")
            break

        print(f"Processing frame of shape: {frame.shape}")
        frame_boxes = []
        results = model.predict(frame, conf=conf, verbose=False)
        if results:
            result = results[0]
            for box in result.boxes:
                x, y, w, h = box.xywh[0]
                cls_idx = box.cls.item()
                pred_conf = box.conf.item()

                x1 = int(x - w / 2)
                y1 = int(y - h / 2)
                x2 = int(x + w / 2)
                y2 = int(y + h / 2)

                bbox_key = (x1, y1, x2, y2)
                frame_boxes.append(bbox_key)

                matched = False
                for tracked_box in bbox_tracker:
                    if is_within_margin(tracked_box, bbox_key, margin=margin):
                        count = bbox_tracker[tracked_box]["count"]
                        bbox_tracker[tracked_box]["count"] = min(count + 1, stability_frames)

                        positions = bbox_tracker[tracked_box]["positions"]
                        if positions.full():
                            positions.get()
                        positions.put((x1, y1, x2, y2))
                        bbox_tracker[tracked_box]["positions"] = positions

                        bbox_tracker[tracked_box]["cls"] = result.names[cls_idx]
                        bbox_tracker[tracked_box]["conf"] = pred_conf

                        matched = True
                        break

                if not matched:
                    positions = Queue(maxsize=position_frames)
                    positions.put((x1, y1, x2, y2))
                    bbox_tracker[bbox_key] = {
                        "count": 1,
                        "positions": positions,
                        "cls": result.names[cls_idx],
                        "conf": pred_conf
                    }
        else:
            print("No results found.")

        for tracked_box in list(bbox_tracker.keys()): # copy keys
            deleted = False
            if not any(is_within_margin(tracked_box, box, margin=margin) for box in frame_boxes):
                bbox_tracker[tracked_box]["count"] -= 1
                if bbox_tracker[tracked_box]["count"] <= 0:
                    del bbox_tracker[tracked_box]
                    deleted = True

            if not deleted and bbox_tracker[tracked_box]["count"] >= stability_frames:
                avg_x1 = int(sum([pos[0] for pos in bbox_tracker[tracked_box]["positions"].queue]) / len(bbox_tracker[tracked_box]["positions"].queue))
                avg_y1 = int(sum([pos[1] for pos in bbox_tracker[tracked_box]["positions"].queue]) / len(bbox_tracker[tracked_box]["positions"].queue))
                avg_x2 = int(sum([pos[2] for pos in bbox_tracker[tracked_box]["positions"].queue]) / len(bbox_tracker[tracked_box]["positions"].queue))
                avg_y2 = int(sum([pos[3] for pos in bbox_tracker[tracked_box]["positions"].queue]) / len(bbox_tracker[tracked_box]["positions"].queue))

                text_x = avg_x1
                text_y = (avg_y1 - 10) if avg_y1 > 10 else (avg_y2 + 30)

                cv2.rectangle(frame, (avg_x1, avg_y1), (avg_x2, avg_y2), (0, 0, 255), 2)
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
    epochs = 10
    batch_size = 16
    img_size = 640
    obb_format = True
    train = True

    dataset_path = os.path.join(os.path.dirname(__file__), "..", "datasets")
    yaml_path = os.path.join(dataset_path, "dice_d6.yaml")
    model_name = "yolov8n-obb" if obb_format else "yolov8n" 

    if train:
        yolo_model = train_model(yaml_path, epochs, batch_size, img_size, device="0", name=model_name)
    else:
        yolo_model = load_model(os.path.join(
            os.path.dirname(__file__),
            "craps-ai",
            "yolov8n4",
            "weights",
            "best.pt"
        ))

    # metrics = yolo_model.val()
    # print(metrics)
    
    stable_predict(
        yolo_model,
        conf=0.5,
        stability_frames=20,
        position_frames=10,
        margin=0.25
    )

    # cap = cv2.VideoCapture(0)
    # while cap.isOpened():
    #     ret, frame = cap.read()
    #     if not ret:
    #         break

    #     results = yolo_model.predict(frame, conf=0.5, verbose=False)
    #     if results:
    #         result = results[0]
    #         for box in result.boxes:
    #             x, y, w, h = box.xywh[0]
    #             cls_idx = box.cls.item()
    #             conf = box.conf.item()

    #             x1 = int(x - w / 2)
    #             y1 = int(y - h / 2)
    #             x2 = int(x + w / 2)
    #             y2 = int(y + h / 2)

    #             cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
    #             cv2.putText(frame, f"{result.names[cls_idx]}, {conf:.2f}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

    #     cv2.imshow("frame", frame)
    #     if cv2.waitKey(1) == ord("q"):
    #         break

    # cap.release()
    # cv2.destroyAllWindows()
