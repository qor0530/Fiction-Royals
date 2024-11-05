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
        delegate=(
            mp.tasks.BaseOptions.Delegate.CPU
            if platform.system() == "Windows"
            else mp.tasks.BaseOptions.Delegate.GPU
        ),
    ),
    num_poses=4,
    running_mode=mp.tasks.vision.RunningMode.IMAGE,
)

# 포즈 랜드마커 객체 생성
landmarker = mp.tasks.vision.PoseLandmarker.create_from_options(options)


def calculate_pose_similarity(landmarks_answer, landmarks_real_time):
    if len(landmarks_answer) != len(landmarks_real_time):
        raise ValueError("리스트 길이 다름")

    flat_answer = np.array(landmarks_answer).flatten()
    flat_real_time = np.array(landmarks_real_time).flatten()

    dot_product = np.dot(flat_answer, flat_real_time)
    norm_answer = np.linalg.norm(flat_answer)
    norm_real_time = np.linalg.norm(flat_real_time)

    if norm_answer == 0 or norm_real_time == 0:
        return 0  # 벡터 크기가 0인 경우 유사도 0 반환

    else:
        cosine_similarity = dot_product / (norm_answer * norm_real_time)  # 0~1

        # 못하면: 0  최고:100
        similarity_percentage = cosine_similarity * 100  # 대략 80~100 사이 값

    return max(
        0, ((similarity_percentage - 80) / 20) * 100
    )  # 유사도 점수 반환 (0~100 범위)


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
