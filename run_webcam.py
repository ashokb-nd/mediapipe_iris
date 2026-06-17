import mediapipe as mp
import cv2
import numpy as np
import os
from datetime import datetime
from threading import Lock
from collections import deque

from iris_module import (
    MODEL_PATH, BaseOptions, FaceLandmarker, FaceLandmarkerOptions,
    VisionRunningMode, draw_landmarks, draw_pupil_position_text,
    compose_frame, HISTORY_MAX,
)


def main():
    latest_result = None
    result_lock = Lock()
    last_timestamp_ms = -1
    writer = None
    output_path = None

    history_l = deque(maxlen=HISTORY_MAX)
    history_r = deque(maxlen=HISTORY_MAX)

    def on_result(result, output_image, timestamp_ms):
        nonlocal latest_result
        with result_lock:
            latest_result = result

    options = FaceLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=MODEL_PATH),
        running_mode=VisionRunningMode.LIVE_STREAM,
        output_face_blendshapes=False,
        num_faces=1,
        result_callback=on_result,
    )

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Cannot open webcam")
        return

    with FaceLandmarker.create_from_options(options) as landmarker:
        print("Press ESC to quit")
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.flip(frame, 1)

            if writer is None:
                fh, fw = frame.shape[:2]
                total_w = fw + 12 + 380 + 12
                right_h = 12 + 220 + 12 + 130 + 12 + 130 + 12
                total_h = max(fh, right_h)
                os.makedirs("DATA/webcam_results", exist_ok=True)
                fps = cap.get(cv2.CAP_PROP_FPS)
                if fps <= 0 or np.isnan(fps):
                    fps = 30.0
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                output_path = os.path.join("DATA/webcam_results", f"webcam_output_{timestamp}.mp4")
                writer = cv2.VideoWriter(
                    output_path,
                    cv2.VideoWriter_fourcc(*"mp4v"),
                    fps,
                    (total_w, total_h),
                )
                if not writer.isOpened():
                    writer = None
                    output_path = None
                    print("Warning: failed to initialize video recording")
                else:
                    print(f"Recording webcam output to: {output_path}")

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            timestamp_ms = int(cap.get(cv2.CAP_PROP_POS_MSEC))
            if timestamp_ms <= 0:
                timestamp_ms = int(cv2.getTickCount() * 1000 / cv2.getTickFrequency())
            if timestamp_ms <= last_timestamp_ms:
                timestamp_ms = last_timestamp_ms + 1
            last_timestamp_ms = timestamp_ms
            landmarker.detect_async(mp_image, timestamp_ms)

            with result_lock:
                result = latest_result

            left_rel = right_rel = None
            left_crop = right_crop = None

            if result and result.face_landmarks:
                draw_landmarks(frame, result.face_landmarks[0])
                left_rel, right_rel, left_crop, right_crop = \
                    draw_pupil_position_text(frame, result.face_landmarks[0])
                if left_rel:
                    history_l.append(left_rel)
                if right_rel:
                    history_r.append(right_rel)

            canvas = compose_frame(frame, left_rel, right_rel,
                                   left_crop, right_crop,
                                   history_l, history_r)

            if writer is not None:
                writer.write(canvas)

            cv2.imshow("Iris Tracking (ESC to quit)", canvas)
            if cv2.waitKey(1) == 27:
                break

    cap.release()
    if writer is not None:
        writer.release()
        print(f"Saved recording: {output_path}")
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
