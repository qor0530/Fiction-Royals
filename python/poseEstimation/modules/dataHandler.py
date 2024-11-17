import os
import numpy as np


def load_dance_database():
    """
    db 디렉토리 내 모든 poses.txt 파일을 읽어 데이터를 반환
    :param db_path: db 디렉토리 경로
    :return: 파싱된 데이터 딕셔너리
    """
    db_path = os.path.join(os.getcwd(), "db")
    result = {}

    for folder_name in os.listdir(db_path):
        folder_path = os.path.join(db_path, folder_name)

        if os.path.isdir(folder_path):
            poses_file_path = os.path.join(folder_path, "poses.txt")

            if os.path.exists(poses_file_path):
                try:
                    with open(poses_file_path, "r") as file:
                        # 첫 번째 줄: 메타 데이터
                        header = file.readline().strip()
                        width, height, frame_interval_raw, video_length_sec = map(
                            float, header.split(",")
                        )

                        # frame_interval이 100.0일 경우 0.1로 간주
                        frame_interval = (
                            0.1 if frame_interval_raw == 100.0 else frame_interval_raw
                        )

                        # 기본 정보 저장
                        data = {
                            "width": int(width),
                            "height": int(height),
                            "fps": int(1 / frame_interval),  # FPS 계산
                            "length": video_length_sec,
                            "landmarks": {},
                        }

                        # 나머지 데이터 처리
                        current_time = 0.0
                        current_2d = []
                        current_3d = []
                        previous_2d = None
                        previous_3d = None

                        for line in file:
                            line = line.strip()

                            if not line:  # 빈 줄: 한 세트 종료
                                if current_2d and current_3d:
                                    current_2d = np.array(current_2d)  # 33x2
                                    left_x, left_y = current_2d[23]
                                    right_x, right_y = current_2d[24]

                                    center_2d = np.array(
                                        [(left_x + right_x) / 2, (left_y + right_y) / 2]
                                    )
                                    current_2d = (current_2d - center_2d).tolist()

                                    # 정상 데이터 저장
                                    data["landmarks"][round(current_time, 1)] = (
                                        current_2d,
                                        current_3d,
                                    )
                                    previous_2d = current_2d
                                    previous_3d = current_3d
                                elif previous_2d and previous_3d:
                                    # None 데이터 처리: 이전 프레임 데이터 복사
                                    data["landmarks"][round(current_time, 1)] = (
                                        previous_2d,
                                        previous_3d,
                                    )

                                current_time += frame_interval
                                current_2d = []
                                current_3d = []
                            else:
                                values = line.split(",")
                                if values == ["None"] * 5:
                                    # None 데이터를 발견한 경우, 건너뜀 (빈 줄에서 처리)
                                    continue
                                elif len(values) == 5:
                                    x_2d, y_2d, x_3d, y_3d, z_3d = map(float, values)
                                    current_2d.append((x_2d, y_2d))
                                    current_3d.append((x_3d, y_3d, z_3d))

                        # 마지막 데이터 처리
                        if current_2d and current_3d:
                            data["landmarks"][round(current_time, 1)] = (
                                current_2d,
                                current_3d,
                            )
                        elif previous_2d and previous_3d:
                            data["landmarks"][round(current_time, 1)] = (
                                previous_2d,
                                previous_3d,
                            )

                        # 결과에 추가
                        result[int(folder_name)] = data

                except Exception as e:
                    print(f"Error reading file {poses_file_path}: {e}")

    return result


def is_keypoint_time(current_time):
    # USG 발표 이후 추후 구현
    return True


def read_answer(database, song_id, current_time):
    # 추후 구현
    print(current_time, "s |", sep="", end=" ")

    width, height = database[song_id]["width"], database[song_id]["height"]

    centered_answer_2D, answer_3D = database[song_id]["landmarks"][current_time]

    return width, height, centered_answer_2D, answer_3D
