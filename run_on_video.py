import argparse

import cv2
import mediapipe as mp

from iris_module import (
    MODEL_PATH, GRAPH_POINTS, BaseOptions, FaceLandmarker,
    FaceLandmarkerOptions, VisionRunningMode, draw_landmarks,
    draw_pupil_position_text, compose_frame, EyeEventDetector,
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
    codec_candidates = [
        ("avc1", "H.264"),
        ("mp4v", "MPEG-4 Part 2"),
    ]
    for fourcc_str, codec_label in codec_candidates:
        fourcc = cv2.VideoWriter_fourcc(*fourcc_str)
        writer = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))
        if writer.isOpened():
            print(f"Using output codec: {codec_label} ({fourcc_str})")
            return writer
        writer.release()

    raise RuntimeError(
        f"Failed to open output video for writing with supported codecs: {output_path}"
    )


def main():
    args = parse_args()
    event_detector = EyeEventDetector()

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
    right_h = 12 + 220 + 12 + 130 + 12 + 130 + 12 + 130 + 12
    total_h = max(frame_height, right_h)

    options = FaceLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=MODEL_PATH),
        running_mode=VisionRunningMode.VIDEO,
        output_face_blendshapes=False,
        num_faces=1,
    )

    # ----------------------------------------------------------------
    # Pass 1: process all frames, store every result
    # ----------------------------------------------------------------
    all_left = []
    all_right = []
    all_avg = []
    all_left_crop = []
    all_right_crop = []

    last_timestamp_ms = -1
    frame_index = 0

    with FaceLandmarker.create_from_options(options) as landmarker:
        print("Pass 1: processing video...")
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

            avg_rel = None
            if left_rel is not None and right_rel is not None:
                avg_rel = ((left_rel[0] + right_rel[0]) / 2,
                           (left_rel[1] + right_rel[1]) / 2)
            all_left.append(left_rel)
            all_right.append(right_rel)
            all_avg.append(avg_rel)
            all_left_crop.append(left_crop)
            all_right_crop.append(right_crop)

            frame_index += 1

        total_frames = frame_index
        print(f"  Processed {total_frames} frames")

    cap.release()

    # ----------------------------------------------------------------
    # Pass 2: re-read video and render with full graph context
    # ----------------------------------------------------------------
    cap = cv2.VideoCapture(args.video_path)

    writer = create_writer(args.output, fps, total_w, total_h) if args.output else None

    half = GRAPH_POINTS // 2
    frame_index = 0

    if not args.no_display:
        print("Pass 2: rendering (ESC to quit)...")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Build centered windows of exactly GRAPH_POINTS entries
        window_l = []
        window_r = []
        window_avg = []
        for j in range(frame_index - half, frame_index + half + 1):
            if 0 <= j < total_frames:
                window_l.append(all_left[j])
                window_r.append(all_right[j])
                window_avg.append(all_avg[j])
            else:
                window_l.append(None)
                window_r.append(None)
                window_avg.append(None)

        history_start = max(0, frame_index - GRAPH_POINTS + 1)
        history_for_state = all_avg[history_start:frame_index + 1]
        current_state = event_detector.current_state(
            history_for_state,
            latest_point=all_avg[frame_index],
        )

        canvas = compose_frame(frame,
                               all_left[frame_index], all_right[frame_index],
                               all_left_crop[frame_index], all_right_crop[frame_index],
                               window_l, window_r, window_avg,
                               event_state=current_state)

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
