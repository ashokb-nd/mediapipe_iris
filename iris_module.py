import cv2
import mediapipe as mp
import numpy as np
from collections import deque

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

RIGHT_WIDTH = 380
GRAPH_W = 340
GRAPH_H = 130
PANEL_W = 290
PANEL_H = 220
PAD = 12
HISTORY_MAX = 500
GRAPH_POINTS = 300
DEFAULT_EVENT_VELOCITY_THRESHOLD = 0.08
DEFAULT_MIN_FIXATION_FRAMES = 5


class EyeEventDetector:
    """
    Identifies fixations and saccades from normalized (x, y) pupil trajectories
    with a velocity-threshold (I-VT) method.
    """

    def __init__(
        self,
        velocity_threshold=DEFAULT_EVENT_VELOCITY_THRESHOLD,
        min_fixation_frames=DEFAULT_MIN_FIXATION_FRAMES,
    ):
        self.velocity_threshold = float(velocity_threshold)
        self.min_fixation_frames = int(min_fixation_frames)

    def detect_events(self, history_buffer):
        valid_data = [pt for pt in history_buffer if pt is not None]
        if len(valid_data) < 2:
            return []

        coords = np.array(valid_data, dtype=np.float32)
        diffs = np.diff(coords, axis=0)
        velocities = np.linalg.norm(diffs, axis=1)

        labels = []
        for v in velocities:
            if v < self.velocity_threshold:
                labels.append("Fixation")
            else:
                labels.append("Saccade")

        if labels:
            labels.append(labels[-1])

        return self._filter_fixations(labels)

    def current_state(self, history_buffer, latest_point=None):
        if latest_point is None:
            return "Unknown"

        labels = self.detect_events(history_buffer)
        return labels[-1] if labels else "Unknown"

    def _filter_fixations(self, labels):
        refined = list(labels)
        n = len(refined)
        i = 0

        while i < n:
            if refined[i] == "Fixation":
                start = i
                while i < n and refined[i] == "Fixation":
                    i += 1
                duration = i - start
                if duration < self.min_fixation_frames:
                    for k in range(start, i):
                        refined[k] = "Saccade"
            else:
                i += 1

        return refined


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


def draw_panel_at(canvas, x, y, left_rel, right_rel, left_crop=None, right_crop=None, event_state="Unknown"):
    panel_w = PANEL_W
    panel_h = PANEL_H

    cv2.rectangle(canvas, (x, y), (x + panel_w, y + panel_h), (30, 30, 30), -1)
    cv2.rectangle(canvas, (x, y), (x + panel_w, y + panel_h), (180, 180, 180), 1)
    cv2.putText(canvas, "Pupil Position", (x + 10, y + 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (220, 220, 220), 1, cv2.LINE_AA)

    box_size = 52
    left_box = (x + 54, y + 34)
    right_box = (x + 184, y + 34)

    for label, (bx, by), rel, color in [
        ("L", left_box, left_rel, (0, 255, 0)),
        ("R", right_box, right_rel, (0, 0, 255)),
    ]:
        cv2.rectangle(canvas, (bx, by), (bx + box_size, by + box_size), (200, 200, 200), 1)
        cv2.putText(canvas, label, (bx + 20, by + box_size + 16), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)

        if rel is not None:
            rx, ry = rel
            dot_x = int(bx + rx * box_size)
            dot_y = int(by + ry * box_size)
            dot_x = max(bx, min(bx + box_size, dot_x))
            dot_y = max(by, min(by + box_size, dot_y))
            cv2.circle(canvas, (dot_x, dot_y), 4, color, -1)

    crop_w, crop_h = EYE_CROP_SIZE
    left_crop_pos = (left_box[0] - 34, left_box[1] + box_size + 24)
    right_crop_pos = (right_box[0] - 34, right_box[1] + box_size + 24)

    cv2.putText(canvas, "Eye Crops", (x + 108, y + 104), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (220, 220, 220), 1, cv2.LINE_AA)

    for label, crop, (cx, cy), color in [
        ("L", left_crop, left_crop_pos, (0, 255, 0)),
        ("R", right_crop, right_crop_pos, (0, 0, 255)),
    ]:
        cv2.rectangle(canvas, (cx - 1, cy - 1), (cx + crop_w + 1, cy + crop_h + 1), (190, 190, 190), 1)
        cv2.putText(canvas, label, (cx - 12, cy + crop_h // 2 + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
        if crop is not None:
            canvas[cy:cy + crop_h, cx:cx + crop_w] = crop
        else:
            cv2.rectangle(canvas, (cx, cy), (cx + crop_w, cy + crop_h), (50, 50, 50), -1)

    state_color = (180, 180, 180)
    if event_state == "Fixation":
        state_color = (80, 230, 80)
    elif event_state == "Saccade":
        state_color = (80, 170, 255)
    cv2.putText(
        canvas,
        f"Event: {event_state}",
        (x + 12, y + panel_h - 10),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.52,
        state_color,
        1,
        cv2.LINE_AA,
    )


def draw_pupil_indicator_panel(image, left_rel, right_rel, left_crop=None, right_crop=None, event_state="Unknown"):
    h, w = image.shape[:2]
    draw_panel_at(image, w - PANEL_W - PAD, PAD, left_rel, right_rel, left_crop, right_crop, event_state)


def draw_pupil_position_text(image, landmarks):
    h, w = image.shape[:2]

    left = pupil_relative_position(landmarks, LEFT_IRIS, LEFT_EYE, LEFT_EYE_OUTER, LEFT_EYE_INNER)
    right = pupil_relative_position(landmarks, RIGHT_IRIS, RIGHT_EYE, RIGHT_EYE_OUTER, RIGHT_EYE_INNER)

    left_rel = (left[0], left[1]) if left else None
    right_rel = (right[0], right[1]) if right else None
    left_crop = extract_eye_crop(image, landmarks, LEFT_EYE)
    right_crop = extract_eye_crop(image, landmarks, RIGHT_EYE)

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

    return left_rel, right_rel, left_crop, right_crop


def draw_history_graph(canvas, x, y, w, h, title, history,
                       color_a, color_b, label_a, label_b):
    bg = (25, 25, 25)
    border = (180, 180, 180)
    text_color = (220, 220, 220)
    grid_color = (50, 50, 50)
    now_color = (200, 200, 0)

    cv2.rectangle(canvas, (x, y), (x + w, y + h), bg, -1)
    cv2.rectangle(canvas, (x, y), (x + w, y + h), border, 1)

    cv2.putText(canvas, title, (x + 8, y + 16), cv2.FONT_HERSHEY_SIMPLEX,
                0.45, text_color, 1, cv2.LINE_AA)

    lx = x + w - 110
    cv2.line(canvas, (lx, y + 8), (lx + 12, y + 8), color_a, 2)
    cv2.putText(canvas, label_a, (lx + 14, y + 12), cv2.FONT_HERSHEY_SIMPLEX,
                0.35, color_a, 1, cv2.LINE_AA)
    cv2.line(canvas, (lx + 50, y + 8), (lx + 62, y + 8), color_b, 2)
    cv2.putText(canvas, label_b, (lx + 64, y + 12), cv2.FONT_HERSHEY_SIMPLEX,
                0.35, color_b, 1, cv2.LINE_AA)

    ml, mr, mt, mb = 36, 10, 22, 10
    px = x + ml
    py = y + mt
    pw = w - ml - mr
    ph = h - mt - mb

    cv2.putText(canvas, "1", (x + 2, py + 8), cv2.FONT_HERSHEY_SIMPLEX,
                0.35, (150, 150, 150), 1, cv2.LINE_AA)
    cv2.putText(canvas, "0", (x + 2, py + ph - 2), cv2.FONT_HERSHEY_SIMPLEX,
                0.35, (150, 150, 150), 1, cv2.LINE_AA)

    for i in range(6):
        gy = int(py + ph * (1 - i / 5))
        cv2.line(canvas, (px, gy), (px + pw, gy), grid_color, 1)

    if not history or len(history) < 2:
        return

    center_idx = GRAPH_POINTS // 2
    pts = list(history)
    m = min(len(pts), GRAPH_POINTS)
    denom = GRAPH_POINTS - 1 if GRAPH_POINTS > 1 else 1

    pts_a = []
    pts_b = []
    for i in range(m):
        val = pts[i]
        if val is None:
            continue
        va, vb = val
        xx = int(px + (i / denom) * pw)
        pts_a.append((xx, int(py + ph * (1 - va))))
        pts_b.append((xx, int(py + ph * (1 - vb))))

    if len(pts_a) >= 2:
        cv2.polylines(canvas, [np.array(pts_a, dtype=np.int32)], False, color_a, 1, cv2.LINE_AA)
        cv2.polylines(canvas, [np.array(pts_b, dtype=np.int32)], False, color_b, 1, cv2.LINE_AA)

    cx = int(px + (center_idx / denom) * pw)
    cv2.line(canvas, (cx, py), (cx, py + ph), now_color, 1)
    cv2.putText(canvas, "now", (cx - 10, py + ph + 8), cv2.FONT_HERSHEY_SIMPLEX,
                0.3, now_color, 1, cv2.LINE_AA)


def compose_frame(video_frame, left_rel, right_rel, left_crop, right_crop,
                  history_l, history_r, history_avg=None, event_state="Unknown"):
    fh, fw = video_frame.shape[:2]

    right_content_h = PAD + PANEL_H + PAD + GRAPH_H + PAD + GRAPH_H + PAD + GRAPH_H + PAD
    total_w = fw + PAD + RIGHT_WIDTH + PAD
    total_h = max(fh, right_content_h)

    canvas = np.zeros((total_h, total_w, 3), dtype=np.uint8)

    canvas[:fh, :fw] = video_frame

    cv2.line(canvas, (fw + PAD - 1, 0), (fw + PAD - 1, total_h - 1),
             (100, 100, 100), 1)

    rx = fw + PAD + (RIGHT_WIDTH - PANEL_W) // 2
    draw_panel_at(canvas, rx, PAD, left_rel, right_rel, left_crop, right_crop, event_state)

    gx = fw + PAD + (RIGHT_WIDTH - GRAPH_W) // 2
    gy1 = PAD + PANEL_H + PAD
    draw_history_graph(canvas, gx, gy1, GRAPH_W, GRAPH_H,
                       "Left Iris Position", history_l,
                       (100, 100, 255), (100, 255, 100), "X", "Y")

    gy2 = gy1 + GRAPH_H + PAD
    draw_history_graph(canvas, gx, gy2, GRAPH_W, GRAPH_H,
                       "Right Iris Position", history_r,
                       (100, 100, 255), (100, 255, 100), "X", "Y")

    gy3 = gy2 + GRAPH_H + PAD
    draw_history_graph(canvas, gx, gy3, GRAPH_W, GRAPH_H,
                       "Avg Iris Position", history_avg or [],
                       (255, 200, 100), (100, 255, 200), "X", "Y")

    return canvas


def draw_landmarks(image, landmarks):
    h, w = image.shape[:2]
    for idx in LEFT_IRIS:
        x, y = int(landmarks[idx].x * w), int(landmarks[idx].y * h)
        r = 2 if idx == LEFT_IRIS_CENTER else 1
        cv2.circle(image, (x, y), r, (0, 255, 0), -1)
    for idx in RIGHT_IRIS:
        x, y = int(landmarks[idx].x * w), int(landmarks[idx].y * h)
        r = 2 if idx == RIGHT_IRIS_CENTER else 1
        cv2.circle(image, (x, y), r, (0, 0, 255), -1)
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
        cv2.circle(image, (x, y), 2, color, -1)
