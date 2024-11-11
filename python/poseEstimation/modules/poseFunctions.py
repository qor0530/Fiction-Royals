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

# 각도 계산을 위한 주요 관절 인덱스 쌍
ANGLE_PAIRS = [
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


def extract_angles(landmarks):
    """지정된 angle_pairs에 따라 landmarks에서 각도를 계산하여 배열로 반환"""
    angles = []
    for i1, i2, i3 in ANGLE_PAIRS:
        angle = calculate_angle(landmarks[i1], landmarks[i2], landmarks[i3])
        if angle is None:
            return None
        else:
            angles.append(angle)
    return np.array(angles)


def calculate_pose_similarity_vectorized(
    normalized_answers_2d, normalized_real_times_2d
):
    """
    여러 사람의 두 포즈 배열에서 각도를 계산한 후 코사인 유사도를 계산하여 0~100 범위로 반환
    """
    # 각 사람별 포즈에서 각도 추출 (여러 명에 대해 적용)
    angles_answers = [extract_angles(ans) for ans in normalized_answers_2d]
    angles_real_times = [extract_angles(rt) for rt in normalized_real_times_2d]

    # 하나라도 None이 있으면 모든 사람의 유사도를 50으로 설정
    if any(angles is None for angles in angles_answers) or any(
        angles is None for angles in angles_real_times
    ):
        return np.full(len(normalized_answers_2d), 50.0)

    # None이 없을 경우 유사도 계산
    angles_answers = np.array(angles_answers)
    angles_real_times = np.array(angles_real_times)

    # 벡터 내적과 크기 계산 (코사인 유사도)
    dot_products = np.einsum("ij,ij->i", angles_answers, angles_real_times)
    norms_answers = np.linalg.norm(angles_answers, axis=1)
    norms_real_times = np.linalg.norm(angles_real_times, axis=1)

    # 코사인 유사도 계산 및 0~100으로 변환
    cosine_similarities = dot_products / (norms_answers * norms_real_times + 1e-5)
    similarity_percentages = ((cosine_similarities - 0.6) / 0.4) * 100
    similarity_percentages = np.clip(similarity_percentages, 0, 100)

    return similarity_percentages.astype(int)


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

    try:
        # 2D 및 3D 랜드마크 추출 및 벡터화
        real_time_landmarks_2d = np.array(
            [
                [(lm.x, lm.y) for lm in landmarks]
                for landmarks in real_time_pose_result.pose_landmarks
            ]
        )

        real_time_landmarks_3d = np.array(
            [
                [(lm.x, lm.y, lm.z) for lm in landmarks]
                for landmarks in real_time_pose_result.pose_world_landmarks
            ]
        )

        if real_time_landmarks_2d.shape[0] == 0:
            return None, None, None, None

        # 중심 좌표 계산 및 정규화
        centers_2d = (real_time_landmarks_2d[:, [23, 24]].mean(axis=1)).reshape(
            -1, 1, 2
        )
        normalized_2d = (real_time_landmarks_2d - centers_2d) * 500 + 250
        normalized_2d = normalized_2d.astype(int)

        # 0번 관절의 x좌표를 기준으로 정렬
        sorted_indices = np.argsort(
            real_time_landmarks_2d[:, 0, 0]
        )  # 0번 관절의 x좌표 기준
        real_time_landmarks_2d_sorted = real_time_landmarks_2d[sorted_indices]
        normalized_2d_sorted = normalized_2d[sorted_indices]
        real_time_landmarks_3d_sorted = real_time_landmarks_3d[sorted_indices]

        return (
            real_time_landmarks_2d_sorted.shape[0],
            real_time_landmarks_2d_sorted,
            normalized_2d_sorted,
            real_time_landmarks_3d_sorted,
        )

    except Exception as e:
        print(f"Error processing frame: {e}")
        return None, None, None, None


def normalize_landmarks(
    pose_landmarks, canvas_size=500, offset=250
) -> tuple[list[tuple[float, float]]]:
    """포즈 랜드마크를 화면의 중심으로 정규화하여 캔버스 크기에 맞게 조정"""
    center = get_center_point(pose_landmarks)

    return [
        (
            int((lm.x - center[0]) * canvas_size) + offset,
            int((lm.y - center[1]) * canvas_size) + offset,
        )
        for lm in pose_landmarks
    ]


def get_center_point(landmarks) -> tuple[float]:
    """골반 인덱스를 기준으로 중심 좌표를 계산"""
    return (
        (landmarks[23].x + landmarks[24].x) / 2,
        (landmarks[23].y + landmarks[24].y) / 2,
    )
