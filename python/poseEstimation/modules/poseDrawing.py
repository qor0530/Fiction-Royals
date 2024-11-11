import cv2


class Painter:
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
    COLORS = [
        (255, 0, 0),  # 빨강
        (0, 255, 0),  # 초록
        (0, 0, 255),  # 파랑
        (255, 255, 0),  # 노랑
        (255, 0, 255),  # 자홍
    ]

    def __init__(self):
        pass

    def draw_realtime_frame(self, frame, human_index, landmarks_2d):
        """이미지에 포즈 랜드마크와 연결선을 그려주는 함수"""
        try:
            color = Painter.COLORS[human_index]

            for u, v in landmarks_2d:
                x, y = int(u * frame.shape[1]), int(v * frame.shape[0])
                cv2.circle(frame, (x, y), 5, (0, 0, 0), -1)

            for start_idx, end_idx in Painter.POSE_CONNECTIONS:
                start_u, start_v = landmarks_2d[start_idx]
                end_u, end_v = landmarks_2d[end_idx]

                start_x = int(start_u * frame.shape[1])
                start_y = int(start_v * frame.shape[0])

                end_x = int(end_u * frame.shape[1])
                end_y = int(end_v * frame.shape[0])
                cv2.line(frame, (start_x, start_y), (end_x, end_y), color, 2)
        except Exception as e:
            print(f"Error drawing landmarks: {e}")

        return frame

    def draw_pose_comparisons(
        self, canvas, human_index, landmarks_answer, landmarks_real_time
    ):
        """정답 포즈와 실시간 포즈 랜드마크를 캔버스에 그려주는 함수"""
        color_answer = Painter.COLORS[0]
        color_realtime = Painter.COLORS[human_index + 1]

        for (x_answer, y_answer), (x_real, y_real) in zip(
            landmarks_answer, landmarks_real_time
        ):
            cv2.circle(canvas, (x_answer, y_answer), 5, (0, 0, 0), -1)
            cv2.circle(canvas, (x_real, y_real), 5, (0, 0, 0), -1)

        for start_idx, end_idx in Painter.POSE_CONNECTIONS:
            cv2.line(
                canvas,
                landmarks_answer[start_idx],
                landmarks_answer[end_idx],
                color_answer,
                2,
            )
            cv2.line(
                canvas,
                landmarks_real_time[start_idx],
                landmarks_real_time[end_idx],
                color_realtime,
                2,
            )

        return canvas


painter = Painter()
print("* 시각화 도구 준비 완료.")
