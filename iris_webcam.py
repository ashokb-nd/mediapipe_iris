import mediapipe as mp
import cv2
import numpy as np

BaseOptions = mp.tasks.BaseOptions
FaceLandmarker = mp.tasks.vision.FaceLandmarker
FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

MODEL_PATH = "face_landmarker.task"

LEFT_IRIS = [468, 469, 470, 471, 472, 473]
RIGHT_IRIS = [474, 475, 476, 477]
LEFT_EYE = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]
RIGHT_EYE = [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398]

def draw_landmarks(image, landmarks):
    h, w = image.shape[:2]
    for idx in LEFT_IRIS:
        x, y = int(landmarks[idx].x * w), int(landmarks[idx].y * h)
        cv2.circle(image, (x, y), 2, (0, 255, 0), -1)
    for idx in RIGHT_IRIS:
        x, y = int(landmarks[idx].x * w), int(landmarks[idx].y * h)
        cv2.circle(image, (x, y), 2, (0, 0, 255), -1)
    for pts, color in [(LEFT_EYE, (255, 255, 0)), (RIGHT_EYE, (255, 255, 0))]:
        pts_c = [(int(landmarks[i].x * w), int(landmarks[i].y * h)) for i in pts]
        for i in range(len(pts_c)):
            cv2.line(image, pts_c[i], pts_c[(i + 1) % len(pts_c)], color, 1)

def main():
    options = FaceLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=MODEL_PATH),
        running_mode=VisionRunningMode.LIVE_STREAM,
        output_face_blendshapes=False,
        num_faces=1,
        result_callback=lambda *args: None,
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
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            result = landmarker.detect(mp_image)

            if result.face_landmarks:
                draw_landmarks(frame, result.face_landmarks[0])

            cv2.imshow("Iris Tracking (ESC to quit)", frame)
            if cv2.waitKey(1) == 27:
                break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
