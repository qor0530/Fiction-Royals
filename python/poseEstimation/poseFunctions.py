import mediapipe as mp
import cv2, platform
import numpy as np

# mp tools
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_pose = mp.solutions.pose

# mp task tools
BaseOptions = mp.tasks.BaseOptions
PoseLandmarker = mp.tasks.vision.PoseLandmarker
PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode


# model load
model_path = "python/poseEstimation/pose_landmarker_heavy.task"

options = PoseLandmarkerOptions(
    base_options=BaseOptions(
        model_asset_path=model_path,
        delegate=(
            BaseOptions.Delegate.CPU
            if platform.system() == "Windows"
            else BaseOptions.Delegate.GPU
        ),
    ),
    num_poses=4,
    running_mode=VisionRunningMode.IMAGE,
)

landmarker = PoseLandmarker.create_from_options(options)

# connections
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


def drawLandmarksOnImage(image, detection_result):
    pose_landmarks_list = detection_result.pose_landmarks

    for pose_landmarks in pose_landmarks_list:
        for landmark in pose_landmarks:
            x, y = int(landmark.x * image.shape[1]), int(landmark.y * image.shape[0])
            cv2.circle(image, (x, y), 5, (0, 255, 0), -1)

        # 랜드마크들 연결하기 (라인 그리기)
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

    return image


def calculatePoseSimilarities(new_landmarks1, new_landmarks2):
    # 두 리스트의 길이가 같아야 함
    if len(new_landmarks1) != len(new_landmarks2):
        raise ValueError("두 랜드마크 리스트의 길이가 같아야 합니다.")

    # 두 리스트를 flatten (1차원 배열로 변환)
    flat_landmarks1 = np.array(new_landmarks1).flatten()
    flat_landmarks2 = np.array(new_landmarks2).flatten()

    # 코사인 유사도 계산
    dot_product = np.dot(flat_landmarks1, flat_landmarks2)
    norm_vec1 = np.linalg.norm(flat_landmarks1)
    norm_vec2 = np.linalg.norm(flat_landmarks2)

    if norm_vec1 == 0 or norm_vec2 == 0:
        cosine_similarity = 0  # 두 벡터 중 하나라도 크기가 0이면 유사도를 0으로 설정
    else:
        cosine_similarity = dot_product / (norm_vec1 * norm_vec2)

    # 백분율로 변환
    similarity_percentage = cosine_similarity * 100  # 일반적으로 80~100 값이 나옴
    final = ((similarity_percentage - 80) / 20) * 100  # 0~100이 나오게 조절

    return max(0, final)  # 음수 처리
