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

    def draw_realtime_frame(self, frame, human_index, real_time_2D):
        try:
            color = Painter.COLORS[human_index]
            width, height, _ = frame.shape

            for u, v in real_time_2D:
                x, y = int(u * height), int(v * width)
                cv2.circle(frame, (x, y), 5, (0, 0, 0), -1)

            for start_idx, end_idx in Painter.POSE_CONNECTIONS:
                start_u, start_v = real_time_2D[start_idx]
                end_u, end_v = real_time_2D[end_idx]

                start_x = int(start_u * height)
                start_y = int(start_v * width)

                end_x = int(end_u * height)
                end_y = int(end_v * width)
                cv2.line(frame, (start_x, start_y), (end_x, end_y), color, 2)
        except Exception as e:
            print(f"Error drawing landmarks: {e}")

        return frame

    def draw_pose_comparisons(
        self, canvas, human_index, centered_answer_2D, centered_real_time_2D
    ):
        color_answer = Painter.COLORS[0]
        color_realtime = Painter.COLORS[human_index]
        w, h, _ = canvas.shape

        for (x_answer, y_answer), (x_real, y_real) in zip(
            centered_answer_2D, centered_real_time_2D
        ):
            x, y = int(x_answer * h + h / 2), int(y_answer * w + w / 2)
            cv2.circle(canvas, (x, y), 5, color_answer, -1)

            x, y = int(x_real * h + h / 2), int(y_real * w + w / 2)
            cv2.circle(canvas, (x, y), 5, color_realtime, -1)

        for start_idx, end_idx in Painter.POSE_CONNECTIONS:
            (answer_start_x, answer_start_y) = centered_answer_2D[start_idx]
            (answer_end_x, answer_end_y) = centered_answer_2D[end_idx]
            (real_time_start_x, real_time_start_y) = centered_real_time_2D[start_idx]
            (real_time_end_x, real_time_end_y) = centered_real_time_2D[end_idx]

            cv2.line(
                canvas,
                (
                    int(answer_start_x * h + h / 2),
                    int(answer_start_y * w + w / 2),
                ),
                (
                    int(answer_end_x * h + h / 2),
                    int(answer_end_y * w + w / 2),
                ),
                color_answer,
                2,
            )
            cv2.line(
                canvas,
                (
                    int(real_time_start_x * h + h / 2),
                    int(real_time_start_y * w + w / 2),
                ),
                (
                    int(real_time_end_x * h + h / 2),
                    int(real_time_end_y * w + w / 2),
                ),
                color_realtime,
                2,
            )

        return canvas


painter = Painter()
print("* 시각화 도구 준비 완료.")
