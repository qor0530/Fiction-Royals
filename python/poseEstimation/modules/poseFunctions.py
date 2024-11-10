import mediapipe as mp
import cv2
import numpy as np
import platform

# 모델 로드 경로 및 옵션 설정
model_path = "python/poseEstimation/models/pose_landmarker_heavy.task"

# GPU 또는 CPU 설정
options = mp.tasks.vision.PoseLandmarkerOptions(
    base_options=mp.tasks.BaseOptions(
        model_asset_path=model_path,
        # model_asset_buffer=open(model_path, "rb").read(),
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
print("* 포즈 추적기 생성 중...")
landmarker = mp.tasks.vision.PoseLandmarker.create_from_options(options)
print("* 포즈 추적기 생성 완료.")


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


def process_frame(frame):
    """카메라 프레임을 처리하여 정답 포즈와 비교할 실시간 랜드마크 좌표 리스트를 정렬하여 반환
    ### n * 2 * (33 * (2 or 3)) 형태.
    ### n은 사람 수
    ### 2는 2D-3D 쌍 튜플(2D, 3D)
    ### 33은 사람 한명 관절 수.
    ### 2 or 3은 픽셀 or 3D 좌표
    """
    real_time_pose_result = landmarker.detect(
        mp.Image(
            image_format=mp.ImageFormat.SRGBA,
            data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGBA),
        )
    )

    landmarks_pairs: list[tuple[list[tuple[float, float]]]] = []

    try:
        for real_time_landmarks_2d, real_time_landmarks_3d in zip(
            real_time_pose_result.pose_landmarks,
            real_time_pose_result.pose_world_landmarks,
        ):
            # 사람 한 명에 대한 데이터로 33 * 2 와 33 * 3 형태(둘 다 list)
            normalized_real_time_2d = normalize_landmarks(real_time_landmarks_2d)
            converted_real_time_landmarks_3d = [
                (lm.x, lm.y, lm.z) for lm in real_time_landmarks_3d
            ]

            landmarks_pairs.append(
                (normalized_real_time_2d, converted_real_time_landmarks_3d)
            )
        return sorted(landmarks_pairs, key=lambda k: k[1][0].x)

    except Exception as e:
        print(f"Error processing frame: {e}")
        return None


def normalize_landmarks(
    pose_landmarks, canvas_size=500, offset=250
) -> tuple[list[tuple[float, float]]]:
    """포즈 랜드마크를 화면의 중심으로 정규화하여 캔버스 크기에 맞게 조정"""
    center = get_center_point(pose_landmarks)

    landmarks = [
        (
            int((lm.x - center[0]) * canvas_size) + offset,
            int((lm.y - center[1]) * canvas_size) + offset,
        )
        for lm in pose_landmarks
    ]

    return landmarks


def get_center_point(landmarks) -> tuple[float]:
    """골반 인덱스를 기준으로 중심 좌표를 계산"""
    center_x = (landmarks[23].x + landmarks[24].x) / 2
    center_y = (landmarks[23].y + landmarks[24].y) / 2
    return center_x, center_y
