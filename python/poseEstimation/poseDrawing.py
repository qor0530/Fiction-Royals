import cv2
import numpy as np

# 랜드마크 연결 정의
POSE_CONNECTIONS = [
    (8, 6),
    (6, 4),
    (4, 0),
    (0, 1),
    (1, 3),
    (3, 7),
    (10, 9),
    (18, 20),
    (18, 16),
    (16, 20),
    (16, 14),
    (14, 12),
    (12, 11),
    (11, 13),
    (13, 15),
    (15, 19),
    (19, 17),
    (17, 15),
    (16, 22),
    (15, 21),
    (12, 24),
    (24, 23),
    (23, 11),
    (24, 26),
    (23, 25),
    (26, 28),
    (25, 27),
    (28, 30),
    (30, 32),
    (32, 28),
    (27, 31),
    (31, 29),
    (29, 27),
]


def draw_landmarks_on_image(image, detection_result):
    """이미지에 포즈 랜드마크와 연결선을 그려주는 함수"""
    try:
        for pose_landmarks in detection_result.pose_landmarks:
            for landmark in pose_landmarks:
                x, y = int(landmark.x * image.shape[1]), int(
                    landmark.y * image.shape[0]
                )
                cv2.circle(image, (x, y), 5, (0, 255, 0), -1)

            for start_idx, end_idx in POSE_CONNECTIONS:
                start_point = pose_landmarks[start_idx]
                end_point = pose_landmarks[end_idx]
                start_x, start_y = int(start_point.x * image.shape[1]), int(
                    start_point.y * image.shape[0]
                )
                end_x, end_y = int(end_point.x * image.shape[1]), int(
                    end_point.y * image.shape[0]
                )
                cv2.line(image, (start_x, start_y), (end_x, end_y), (255, 0, 0), 2)
    except Exception as e:
        print(f"Error drawing landmarks: {e}")

    return image


def draw_answer_pose_on_canvas(
    landmarks_answer, landmarks_real_time, canvas_size=(500, 500)
):
    """정답 포즈와 실시간 포즈 랜드마크를 캔버스에 그려주는 함수"""
    canvas = (
        np.ones((canvas_size[0], canvas_size[1], 3), dtype=np.uint8) * 255
    )  # 흰색 배경 생성

    # 각 랜드마크와 연결선을 그린다
    for (x_answer, y_answer), (x_real, y_real) in zip(
        landmarks_answer, landmarks_real_time
    ):
        cv2.circle(
            canvas, (x_answer, y_answer), 5, (0, 255, 0), -1
        )  # 정답 포즈의 랜드마크
        cv2.circle(
            canvas, (x_real, y_real), 5, (0, 255, 0), -1
        )  # 실시간 포즈의 랜드마크
    for start_idx, end_idx in POSE_CONNECTIONS:
        cv2.line(
            canvas,
            landmarks_answer[start_idx],
            landmarks_answer[end_idx],
            (255, 0, 0),
            2,
        )  # 정답 포즈 연결선
        cv2.line(
            canvas,
            landmarks_real_time[start_idx],
            landmarks_real_time[end_idx],
            (255, 0, 0),
            2,
        )  # 실시간 포즈 연결선

    return True, canvas
