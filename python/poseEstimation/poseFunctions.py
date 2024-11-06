import mediapipe as mp
import cv2
import numpy as np
import platform

# 모델 로드 경로 및 옵션 설정
model_path = "python/poseEstimation/pose_landmarker_heavy.task"

# GPU 또는 CPU 설정
options = mp.tasks.vision.PoseLandmarkerOptions(
    base_options=mp.tasks.BaseOptions(
        model_asset_path=model_path,
        delegate=mp.tasks.BaseOptions.Delegate.CPU,
    ),
    num_poses=4,
    running_mode=mp.tasks.vision.RunningMode.IMAGE,
)

# 포즈 랜드마커 객체 생성
landmarker = mp.tasks.vision.PoseLandmarker.create_from_options(options)


# 이 함수 담당!

# 각도 계산을 위한 주요 관절 인덱스 쌍
angle_pairs = [
    (11, 13, 15),  # 오른쪽 팔
    (12, 14, 16),  # 왼쪽 팔
    (25, 27, 29),  # 오른쪽 다리
    (26, 28, 30),  # 왼쪽 다리
    (11, 23, 25),  # 오른쪽 몸통
    (12, 24, 26),  # 왼쪽 몸통
    (13, 11, 23),  # 오른쪽 겨드랑이 (11번 기준 13, 23)
    (14, 12, 24),  # 왼쪽 겨드랑이 (12번 기준 14, 24)
    (23, 24, 26),  # 24번 기준으로 23, 26 각도
    (24, 23, 25),  # 23번 기준으로 24, 25 각도
]

def calculate_angle(p1, p2, p3):
    """세 점을 이용해 p2에서 p1, p3 사이의 각도를 계산하고 반환 (도 단위)"""
    u = [p1[0] - p2[0], p1[1] - p2[1]]
    v = [p3[0] - p2[0], p3[1] - p2[1]]

    dot_product = np.dot(u, v)
    u_norm = np.linalg.norm(u)
    v_norm = np.linalg.norm(v)

    if u_norm == 0 or v_norm == 0:
        return None

    cos_theta = dot_product / (u_norm * v_norm)
    angle = np.arccos(np.clip(cos_theta, -1.0, 1.0))
    angle_degrees = np.degrees(angle)

    return angle_degrees


def extract_angles(landmarks, angle_pairs):
    """지정된 angle_pairs에 따라 landmarks에서 각도를 계산하여 배열로 반환"""
    angles = []
    for (i1, i2, i3) in angle_pairs:
        angle = calculate_angle(landmarks[i1], landmarks[i2], landmarks[i3])
        if angle is not None:
            angles.append(angle)
    return np.array(angles)


def calculate_pose_similarity(angles_answer, angles_real_time):
    """두 각도 배열의 코사인 유사도를 계산하여 0~100 범위로 반환"""
    dot_product = np.dot(angles_answer, angles_real_time)
    norm_answer = np.linalg.norm(angles_answer)
    norm_real_time = np.linalg.norm(angles_real_time)

    if norm_answer == 0 or norm_real_time == 0:
        return 0  # 벡터 크기가 0인 경우 유사도 0 반환

    cosine_similarity = dot_product / (norm_answer * norm_real_time)  # 0~1 범위
    return cosine_similarity * 100  # %로 변환하여 반환


#################


def process_frame(frame, answer_pose_landmarks):
    """카메라 프레임을 처리하여 정답 포즈와 비교할 실시간 랜드마크 좌표 리스트를 반환"""
    real_time_pose_result = landmarker.detect(
        mp.Image(
            image_format=mp.ImageFormat.SRGBA,
            data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGBA),
        )
    )

    landmarks_pairs = []
    try:
        for answer_landmarks, real_time_landmarks in zip(
            answer_pose_landmarks, real_time_pose_result.pose_landmarks
        ):
            normalized_answer, normalized_real_time = normalize_landmarks(
                answer_landmarks, real_time_landmarks
            )
            landmarks_pairs.append((normalized_answer, normalized_real_time))
        return landmarks_pairs
    except Exception as e:
        print(f"Error processing frame: {e}")
        return None


def normalize_landmarks(
    pose_landmarks_answer, pose_landmarks_real_time, canvas_size=500, offset=250
):
    """정답 포즈와 실시간 포즈 랜드마크를 중심으로 정규화하여 캔버스 크기에 맞게 조정"""
    center_answer = get_center_point(pose_landmarks_answer, [23, 24])
    center_real_time = get_center_point(pose_landmarks_real_time, [23, 24])

    landmarks_answer = [
        (
            int((lm.x - center_answer[0]) * canvas_size) + offset,
            int((lm.y - center_answer[1]) * canvas_size) + offset,
        )
        for lm in pose_landmarks_answer
    ]
    landmarks_real_time = [
        (
            int((lm.x - center_real_time[0]) * canvas_size) + offset,
            int((lm.y - center_real_time[1]) * canvas_size) + offset,
        )
        for lm in pose_landmarks_real_time
    ]

    return landmarks_answer, landmarks_real_time


def get_center_point(landmarks, indices):
    """특정 랜드마크 인덱스를 기준으로 중심 좌표를 계산"""
    return (
        sum(landmarks[i].x for i in indices) / len(indices),
        sum(landmarks[i].y for i in indices) / len(indices),
    )

