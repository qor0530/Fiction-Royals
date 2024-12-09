import mediapipe as mp
import cv2
import numpy as np
import platform

model_path = "python/poseEstimation/models/pose_landmarker_heavy.task"

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

print("* 포즈 추적기 생성 중...")
landmarker = mp.tasks.vision.PoseLandmarker.create_from_options(options)
print(platform.system())
print("* 포즈 추적기 생성 완료.")


ANGLE_PAIRS = [
    (11, 13, 15),
    (12, 14, 16),
    (25, 27, 29),
    (26, 28, 30),
    (11, 23, 25),
    (12, 24, 26),
    (13, 11, 23),
    (14, 12, 24),
    (23, 24, 26),
    (24, 23, 25),
]


def extract_angles(landmarks):
    def calculate_angle(p1, p2, p3):
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

    angles = []
    for i1, i2, i3 in ANGLE_PAIRS:
        angle = calculate_angle(landmarks[i1], landmarks[i2], landmarks[i3])
        if angle is None:
            raise Exception("Angle Error")
        else:
            angles.append(angle)
    return np.array(angles)


def calculate_pose_similarity_vectorized(
    num_of_humans, centered_answer_2D, sorted_centered_2Ds
):
    try:
        angles_answers = np.array(
            [extract_angles(centered_answer_2D)] * num_of_humans)
        angles_real_times = np.array(
            [extract_angles(one_person_2D)
             for one_person_2D in sorted_centered_2Ds]
        )
    except:
        return np.full(num_of_humans, 50.0)

    dot_products = np.einsum("ij,ij->i", angles_answers, angles_real_times)
    norms_answers = np.linalg.norm(angles_answers, axis=1)
    norms_real_times = np.linalg.norm(angles_real_times, axis=1)

    cosine_similarities = dot_products / \
        (norms_answers * norms_real_times + 1e-5)
    similarity_percentages = ((cosine_similarities - 0.6) / 0.4) * 100
    similarity_percentages = np.clip(similarity_percentages, 0, 100)

    return similarity_percentages.astype(int)


def process_frame(frame):
    """카메라 프레임을 처리하여 정렬된 랜드마크 반환"""
    real_time_pose_result = landmarker.detect(
        mp.Image(
            image_format=mp.ImageFormat.SRGBA,
            data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGBA),
        )
    )

    if not real_time_pose_result.pose_landmarks:
        return 0, None, None, None

    try:
        real_time_2Ds = np.array(
            [
                [(lm.x, lm.y) for lm in landmarks]
                for landmarks in real_time_pose_result.pose_landmarks
            ]
        )

        real_time_3Ds = np.array(
            [
                [(lm.x, lm.y, lm.z) for lm in landmarks]
                for landmarks in real_time_pose_result.pose_world_landmarks
            ]
        )

        centers_2Ds = (real_time_2Ds[:, [23, 24]].mean(
            axis=1)).reshape(-1, 1, 2)
        centered_real_time_2Ds = real_time_2Ds - centers_2Ds

        # 0번 관절(머리)의 x좌표를 기준으로 정렬
        sorted_indices = np.argsort(real_time_2Ds[:, 0, 0])
        sorted_real_time_2Ds = real_time_2Ds[sorted_indices]
        sorted_real_time_3Ds = real_time_3Ds[sorted_indices]
        sorted_centered_real_time_2Ds = centered_real_time_2Ds[sorted_indices]

        return (
            sorted_real_time_2Ds.shape[0],
            sorted_real_time_2Ds,
            sorted_centered_real_time_2Ds,
            sorted_real_time_3Ds,
        )

    except Exception as e:
        print(f"Error processing frame: {e}")
        return 0, None, None, None
