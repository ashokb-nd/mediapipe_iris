import cv2
import mediapipe as mp
import numpy as np

BaseOptions = mp.tasks.BaseOptions
FaceLandmarker = mp.tasks.vision.FaceLandmarker
FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

MODEL_PATH = "DATA/face_landmarker.task"

LEFT_IRIS = [468, 469, 470, 471, 472]
RIGHT_IRIS = [473, 474, 475, 476, 477]
LEFT_IRIS_CENTER = 468
RIGHT_IRIS_CENTER = 473
LEFT_EYE = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]
RIGHT_EYE = [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398]
LEFT_EYE_OUTER = 33
LEFT_EYE_INNER = 133
RIGHT_EYE_OUTER = 362
RIGHT_EYE_INNER = 263

EYE_CROP_SIZE = (120, 68)
EYE_CROP_PAD = 12


def pupil_relative_position(landmarks, iris_indices, eye_indices, outer_corner_idx, inner_corner_idx):
    iris_x = np.mean([landmarks[i].x for i in iris_indices])
    iris_y = np.mean([landmarks[i].y for i in iris_indices])

    outer = np.array([landmarks[outer_corner_idx].x, landmarks[outer_corner_idx].y], dtype=np.float32)
    inner = np.array([landmarks[inner_corner_idx].x, landmarks[inner_corner_idx].y], dtype=np.float32)
    axis = inner - outer
    axis_len = float(np.linalg.norm(axis))
    if axis_len <= 1e-6:
        return None

    x_hat = axis / axis_len
    y_hat = np.array([-x_hat[1], x_hat[0]], dtype=np.float32)

    eye_pts = [np.array([landmarks[i].x, landmarks[i].y], dtype=np.float32) for i in eye_indices]
    eye_u = [float(np.dot(p - outer, x_hat)) for p in eye_pts]
    eye_v = [float(np.dot(p - outer, y_hat)) for p in eye_pts]
    min_u, max_u = min(eye_u), max(eye_u)
    min_v, max_v = min(eye_v), max(eye_v)

    width = max_u - min_u
    height = max_v - min_v
    if width <= 1e-6 or height <= 1e-6:
        return None

    iris_pt = np.array([iris_x, iris_y], dtype=np.float32)
    iris_u = float(np.dot(iris_pt - outer, x_hat))
    iris_v = float(np.dot(iris_pt - outer, y_hat))

    rel_x = (iris_u - min_u) / width
    rel_y = (iris_v - min_v) / height
    rel_x = float(np.clip(rel_x, 0.0, 1.0))
    rel_y = float(np.clip(rel_y, 0.0, 1.0))
    return rel_x, rel_y, iris_x, iris_y


def extract_eye_crop(image, landmarks, eye_indices, pad=EYE_CROP_PAD, out_size=EYE_CROP_SIZE):
    h, w = image.shape[:2]
    pts = np.array([(int(landmarks[i].x * w), int(landmarks[i].y * h)) for i in eye_indices], dtype=np.int32)
    if pts.size == 0:
        return None

    x, y, bw, bh = cv2.boundingRect(pts)
    x0 = max(0, x - pad)
    y0 = max(0, y - pad)
    x1 = min(w, x + bw + pad)
    y1 = min(h, y + bh + pad)
    if x1 <= x0 or y1 <= y0:
        return None

    crop = image[y0:y1, x0:x1]
    if crop.size == 0:
        return None
    return cv2.resize(crop, out_size, interpolation=cv2.INTER_LINEAR)


def draw_pupil_indicator_panel(image, left_rel, right_rel, left_crop=None, right_crop=None):
    h, w = image.shape[:2]
    panel_w = 290
    panel_h = 220
    pad = 12
    panel_x = w - panel_w - pad
    panel_y = pad

    cv2.rectangle(image, (panel_x, panel_y), (panel_x + panel_w, panel_y + panel_h), (30, 30, 30), -1)
    cv2.rectangle(image, (panel_x, panel_y), (panel_x + panel_w, panel_y + panel_h), (180, 180, 180), 1)
    cv2.putText(image, "Pupil Position", (panel_x + 10, panel_y + 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (220, 220, 220), 1, cv2.LINE_AA)

    box_size = 52
    left_box = (panel_x + 54, panel_y + 34)
    right_box = (panel_x + 184, panel_y + 34)

    for label, (bx, by), rel, color in [
        ("L", left_box, left_rel, (0, 255, 0)),
        ("R", right_box, right_rel, (0, 0, 255)),
    ]:
        cv2.rectangle(image, (bx, by), (bx + box_size, by + box_size), (200, 200, 200), 1)
        cv2.putText(image, label, (bx + 20, by + box_size + 16), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)

        if rel is not None:
            rx, ry = rel
            dot_x = int(bx + rx * box_size)
            dot_y = int(by + ry * box_size)
            dot_x = max(bx, min(bx + box_size, dot_x))
            dot_y = max(by, min(by + box_size, dot_y))
            cv2.circle(image, (dot_x, dot_y), 4, color, -1)

    crop_w, crop_h = EYE_CROP_SIZE
    left_crop_pos = (left_box[0] - 34, left_box[1] + box_size + 24)
    right_crop_pos = (right_box[0] - 34, right_box[1] + box_size + 24)

    cv2.putText(image, "Eye Crops", (panel_x + 108, panel_y + 104), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (220, 220, 220), 1, cv2.LINE_AA)

    for label, crop, (cx, cy), color in [
        ("L", left_crop, left_crop_pos, (0, 255, 0)),
        ("R", right_crop, right_crop_pos, (0, 0, 255)),
    ]:
        cv2.rectangle(image, (cx - 1, cy - 1), (cx + crop_w + 1, cy + crop_h + 1), (190, 190, 190), 1)
        cv2.putText(image, label, (cx - 12, cy + crop_h // 2 + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
        if crop is not None:
            image[cy:cy + crop_h, cx:cx + crop_w] = crop
        else:
            cv2.rectangle(image, (cx, cy), (cx + crop_w, cy + crop_h), (50, 50, 50), -1)


def draw_pupil_position_text(image, landmarks):
    h, w = image.shape[:2]

    left = pupil_relative_position(landmarks, LEFT_IRIS, LEFT_EYE, LEFT_EYE_OUTER, LEFT_EYE_INNER)
    right = pupil_relative_position(landmarks, RIGHT_IRIS, RIGHT_EYE, RIGHT_EYE_OUTER, RIGHT_EYE_INNER)

    left_rel = (left[0], left[1]) if left else None
    right_rel = (right[0], right[1]) if right else None
    left_crop = extract_eye_crop(image, landmarks, LEFT_EYE)
    right_crop = extract_eye_crop(image, landmarks, RIGHT_EYE)
    draw_pupil_indicator_panel(image, left_rel, right_rel, left_crop, right_crop)

    if left:
        lrx, lry, _, _ = left
        lix = landmarks[LEFT_IRIS_CENTER].x
        liy = landmarks[LEFT_IRIS_CENTER].y
        cv2.putText(
            image,
            f"L pupil (x,y): ({lrx:.2f}, {lry:.2f})",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )
        cv2.circle(image, (int(lix * w), int(liy * h)), 4, (0, 255, 0), -1)

    if right:
        rrx, rry, _, _ = right
        rix = landmarks[RIGHT_IRIS_CENTER].x
        riy = landmarks[RIGHT_IRIS_CENTER].y
        cv2.putText(
            image,
            f"R pupil (x,y): ({rrx:.2f}, {rry:.2f})",
            (10, 55),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 0, 255),
            2,
            cv2.LINE_AA,
        )
        cv2.circle(image, (int(rix * w), int(riy * h)), 4, (0, 0, 255), -1)


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

    anchor_points = [
        (LEFT_EYE_OUTER, (0, 255, 255)),
        (LEFT_EYE_INNER, (0, 200, 255)),
        (RIGHT_EYE_OUTER, (255, 0, 255)),
        (RIGHT_EYE_INNER, (255, 100, 255)),
    ]
    for idx, color in anchor_points:
        x, y = int(landmarks[idx].x * w), int(landmarks[idx].y * h)
        cv2.circle(image, (x, y), 5, color, -1)
