import time
import cv2
from ultralytics import YOLO

# from sos import alert

def throttle(seconds):
    def decorator(func):
        last_called = 0

        def wrapper(*args, **kwargs):
            nonlocal last_called
            current_time = time.time()
            if current_time - last_called < seconds:
                return None
            else:
                last_called = current_time
                return func(*args, **kwargs)
        return wrapper
    return decorator


@throttle(10)
def trigger():
    # alert()
    pass


def detect_objects_in_video(video_path):
    yolo_model = YOLO('./weights/norm.pt')
    video_capture = cv2.VideoCapture(video_path)

    while True:
        ret, frame = video_capture.read()
        if not ret:
            break
        results = yolo_model(frame)

        for result in results:
            classes = result.names
            
            cls = result.boxes.cls
            conf = result.boxes.conf
            detections = result.boxes.xyxy
            for pos, detection in enumerate(detections):
                if conf[pos] >= 0.5:
                    trigger()
                    xmin, ymin, xmax, ymax = detection
                    label = f"{classes[int(cls[pos])]} {conf[pos]:.2f}" 
                    color = (0, int(cls[pos]), 255)
                    cv2.rectangle(frame, (int(xmin), int(ymin)), (int(xmax), int(ymax)), color, 2)
                    cv2.putText(frame, label, (int(xmin), int(ymin) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
        cv2.imshow('Object Detection',frame)
        if cv2.waitKey(1) & 0xFF == ord('q'): 
            break
    cv2.destroyAllWindows()


detect_objects_in_video(0)


