import mediapipe as mp
import cv2
import numpy as np

BaseOptions = mp.tasks.BaseOptions
FaceLandmarker = mp.tasks.vision.FaceLandmarker
FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

MODEL_PATH = "DATA/face_landmarker.task"

LEFT_IRIS = [468, 469, 470, 471, 472, 473]
RIGHT_IRIS = [474, 475, 476, 477]
LEFT_EYE = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]
RIGHT_EYE = [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398]

IRIS_DIAMETER_MM = 11.8

def iris_diameter_px(landmarks, iris_indices, w):
    pts = [(int(landmarks[i].x * w), int(landmarks[i].y * landmarks[0].z * 0 + landmarks[i].z)) for i in iris_indices]
    xs = [landmarks[i].x * w for i in iris_indices]
    return max(xs) - min(xs)

def estimate_depth(landmarks, focal_length_px, image_width):
    left_diameter = iris_diameter_px(landmarks, LEFT_IRIS, image_width)
    right_diameter = iris_diameter_px(landmarks, RIGHT_IRIS, image_width)
    avg_diameter_px = (left_diameter + right_diameter) / 2
    if avg_diameter_px < 1:
        return None
    depth_mm = (focal_length_px * IRIS_DIAMETER_MM) / avg_diameter_px
    return depth_mm / 10.0

def draw_iris(image, landmarks):
    h, w = image.shape[:2]
    for idx in LEFT_IRIS:
        x, y = int(landmarks[idx].x * w), int(landmarks[idx].y * h)
        cv2.circle(image, (x, y), 2, (0, 255, 0), -1)
    for idx in RIGHT_IRIS:
        x, y = int(landmarks[idx].x * w), int(landmarks[idx].y * h)
        cv2.circle(image, (x, y), 2, (0, 0, 255), -1)

    for idx in [468, 473]:
        cx, cy = int(landmarks[idx].x * w), int(landmarks[idx].y * h)
        cv2.circle(image, (cx, cy), 4, (255, 255, 0), 2)

def draw_eye_contours(image, landmarks):
    h, w = image.shape[:2]
    for eye_pts, color in [(LEFT_EYE, (255, 255, 0)), (RIGHT_EYE, (255, 255, 0))]:
        pts = [(int(landmarks[i].x * w), int(landmarks[i].y * h)) for i in eye_pts]
        for i in range(len(pts)):
            cv2.line(image, pts[i], pts[(i + 1) % len(pts)], color, 1)

def draw_face_outline(image, landmarks):
    h, w = image.shape[:2]
    contour = [10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 397, 365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136, 172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109, 10]
    pts = [(int(landmarks[i].x * w), int(landmarks[i].y * h)) for i in contour]
    for i in range(len(pts) - 1):
        cv2.line(image, pts[i], pts[i + 1], (200, 200, 200), 1)

def process_image(image_path, focal_length_px=None):
    options = FaceLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=MODEL_PATH),
        running_mode=VisionRunningMode.IMAGE,
        num_faces=1,
    )

    with FaceLandmarker.create_from_options(options) as landmarker:
        image = cv2.imread(image_path)
        if image is None:
            print(f"Failed to load: {image_path}")
            return

        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        result = landmarker.detect(mp_image)

        if result.face_landmarks:
            for fl in result.face_landmarks:
                draw_face_outline(image, fl)
                draw_eye_contours(image, fl)
                draw_iris(image, fl)
                lx, ly = fl[468].x, fl[468].y
                rx, ry = fl[473].x, fl[473].y
                print(f"Left iris:  ({lx:.3f}, {ly:.3f})")
                print(f"Right iris: ({rx:.3f}, {ry:.3f})")
                if focal_length_px:
                    d = estimate_depth(fl, focal_length_px, image.shape[1])
                    if d:
                        print(f"Estimated distance: {d:.1f} cm")
        else:
            print("No face detected")

        out = "output_" + image_path.rsplit("/", 1)[-1]
        cv2.imwrite(out, image)
        print(f"Saved: {out}")

if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python iris_demo.py <image_path> [focal_length_px]")
        print("Example: python iris_demo.py face.jpg 1200")
    else:
        fl = float(sys.argv[2]) if len(sys.argv) > 2 else None
        process_image(sys.argv[1], fl)
