import cv2
import mediapipe as mp
import numpy as np

# Mediapipe 초기화
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=True, model_complexity=2)
mp_drawing = mp.solutions.drawing_utils

# 각도 계산을 위한 관절 인덱스 쌍
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

def draw_angle_on_image(image, landmarks, angle_pairs):
    """이미지에 각도 계산 후 표시"""
    for i, (i1, i2, i3) in enumerate(angle_pairs):
        p1, p2, p3 = landmarks[i1], landmarks[i2], landmarks[i3]
        angle = calculate_angle(p1, p2, p3)

        if angle is not None:
            # 각도 값을 이미지에 표시
            pos_x, pos_y = int(p2[0] * image.shape[1]), int(p2[1] * image.shape[0])
            cv2.putText(
                image,
                f'{int(angle)}°',
                (pos_x, pos_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2,
                cv2.LINE_AA
            )

def get_landmarks_from_image(image_path):
    """이미지에서 포즈 랜드마크를 추출"""
    image = cv2.imread(image_path)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = pose.process(image_rgb)

    if not results.pose_landmarks:
        print("포즈 랜드마크를 감지하지 못했습니다.")
        return None, None

    landmarks = [(lm.x, lm.y) for lm in results.pose_landmarks.landmark]
    return image, landmarks, results

def display_image_with_angles(image_path):
    """이미지에 각도를 표시하여 보여줌"""
    image, landmarks, results = get_landmarks_from_image(image_path)
    if image is None or landmarks is None:
        return

    # 각도를 계산하고 이미지에 표시
    draw_angle_on_image(image, landmarks, angle_pairs)

    # 랜드마크와 연결선 그리기
    mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

    # 결과 이미지 표시
    cv2.imshow("Image with Angles", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# 실행 예시
image_path = "/Users/jeongsieun/Fiction-Royals/python/poseEstimation/IMG_1742.jpg"
display_image_with_angles(image_path)
