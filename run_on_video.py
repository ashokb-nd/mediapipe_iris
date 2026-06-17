import argparse

import cv2
import mediapipe as mp
from collections import deque

from iris_module import (
    MODEL_PATH, BaseOptions, FaceLandmarker, FaceLandmarkerOptions,
    VisionRunningMode, draw_landmarks, draw_pupil_position_text,
    compose_frame, HISTORY_MAX,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Run iris tracking on a video file.")
    parser.add_argument("video_path", help="Path to the input video file")
    parser.add_argument("--output", help="Optional path to save the annotated video")
    parser.add_argument(
        "--no-display",
        action="store_true",
        help="Process without opening a preview window",
    )
    return parser.parse_args()


def create_writer(output_path, fps, frame_width, frame_height):
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))
    if not writer.isOpened():
        raise RuntimeError(f"Failed to open output video for writing: {output_path}")
    return writer


def main():
    args = parse_args()

    cap = cv2.VideoCapture(args.video_path)
    if not cap.isOpened():
        print(f"Cannot open video: {args.video_path}")
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 30.

    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    total_w = frame_width + 12 + 380 + 12
    right_h = 12 + 220 + 12 + 130 + 12 + 130 + 12
    total_h = max(frame_height, right_h)

    writer = create_writer(args.output, fps, total_w, total_h) if args.output else None

    options = FaceLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=MODEL_PATH),
        running_mode=VisionRunningMode.VIDEO,
        output_face_blendshapes=False,
        num_faces=1,
    )

    last_timestamp_ms = -1
    frame_index = 0
    history_l = deque(maxlen=HISTORY_MAX)
    history_r = deque(maxlen=HISTORY_MAX)

    with FaceLandmarker.create_from_options(options) as landmarker:
        if not args.no_display:
            print("Press ESC to quit")

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

            timestamp_ms = int(cap.get(cv2.CAP_PROP_POS_MSEC))
            if timestamp_ms <= 0:
                timestamp_ms = int(frame_index * 1000 / fps)
            if timestamp_ms <= last_timestamp_ms:
                timestamp_ms = last_timestamp_ms + 1
            last_timestamp_ms = timestamp_ms

            result = landmarker.detect_for_video(mp_image, timestamp_ms)

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

            if not args.no_display:
                cv2.imshow("Iris Tracking (ESC to quit)", canvas)
                key = cv2.waitKey(1)
                if key == 27:
                    break

            frame_index += 1

    cap.release()
    if writer is not None:
        writer.release()
        print(f"Saved annotated video to: {args.output}")
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
