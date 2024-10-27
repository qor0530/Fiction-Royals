import cv2
import numpy as np
from poseFunctions import *
from unityCommunication import *
from dataHandler import *


def draw_answer_pose_on_canvas(lm1, lm2, canvas_size=(500, 500)):
    canvas = (
        np.ones((canvas_size[0], canvas_size[1], 3), dtype=np.uint8) * 255
    )  # 흰색 배경

    # 랜드마크 그리기
    for (x1, y1), (x2, y2) in zip(lm1, lm2):
        cv2.circle(canvas, (x1, y1), 5, (0, 255, 0), -1)  # 녹색 점 그리기
        cv2.circle(canvas, (x2, y2), 5, (0, 255, 0), -1)  # 녹색 점 그리기

    # 랜드마크들 연결하기 (라인 그리기)
    for s, e in POSE_CONNECTIONS:  # POSE_CONNECTIONS는 미리 정의된 랜드마크 연결 쌍
        sx1, sy1 = lm1[s]
        ex1, ey1 = lm1[e]
        sx2, sy2 = lm2[s]
        ex2, ey2 = lm2[e]

        cv2.line(canvas, (sx1, sy1), (ex1, ey1), (255, 0, 0), 2)
        cv2.line(canvas, (sx2, sy2), (ex2, ey2), (255, 0, 0), 2)

    return True, canvas


def scale_landmarks(landmarks, target_min=100, target_max=400):
    # 각 랜드마크의 x, y 최소, 최대값 계산
    min_x = min([x for x, y in landmarks])
    max_x = max([x for x, y in landmarks])
    min_y = min([y for x, y in landmarks])
    max_y = max([y for x, y in landmarks])

    # x, y 좌표를 100~400 사이로 스케일링
    scaled_landmarks = []
    for x, y in landmarks:
        scaled_x = target_min + (x - min_x) * (target_max - target_min) / (
            max_x - min_x
        )
        scaled_y = target_min + (y - min_y) * (target_max - target_min) / (
            max_y - min_y
        )
        scaled_landmarks.append((int(scaled_x), int(scaled_y)))

    return scaled_landmarks


# 영상 및 정답 DB 로딩
key_points = loadDanceDatabase()
answer_img = cv2.imread("python/poseEstimation/dance.jpg")
answer_pose = landmarker.detect(
    mp.Image(
        image_format=mp.ImageFormat.SRGBA,
        data=cv2.cvtColor(answer_img, cv2.COLOR_BGR2RGBA),
    )
)

# 카메라 초기화
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("카메라를 여는데 실패했습니다.")
    exit(0)

print("======================================================================")

while True:
    # -------------- 카메라 처리 --------------#
    ret, frame = cap.read()
    if not ret:
        break

    # -------------- 포즈 처리 --------------#
    pose_landmarker_result = landmarker.detect(
        mp.Image(
            image_format=mp.ImageFormat.SRGBA,
            data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGBA),
        )
    )

    try:
        pose_landmarks1 = answer_pose.pose_landmarks[0]
        pose_landmarks2 = pose_landmarker_result.pose_landmarks[0]

        center1_x = (pose_landmarks1[23].x + pose_landmarks1[24].x) / 2
        center1_y = (pose_landmarks1[23].y + pose_landmarks1[24].y) / 2
        center2_x = (pose_landmarks2[23].x + pose_landmarks2[24].x) / 2
        center2_y = (pose_landmarks2[23].y + pose_landmarks2[24].y) / 2

        new_landmarks1, new_landmarks2 = [], []
        for lm1, lm2 in zip(pose_landmarks1, pose_landmarks2):
            x1, y1 = (
                int((lm1.x - center1_x) * 500) + 250,
                int((lm1.y - center1_y) * 500) + 250,
            )

            x2, y2 = (
                int((lm2.x - center2_x) * 500) + 250,
                int((lm2.y - center2_y) * 500) + 250,
            )

            new_landmarks1.append((x1, y1))
            new_landmarks2.append((x2, y2))

    except:
        continue

    # 정규화해서 화면 안에 맞추기(성능 잘 안나오는 듯)
    # new_landmarks1 = scale_landmarks(new_landmarks1)
    # new_landmarks2 = scale_landmarks(new_landmarks2)

    # -------------- 유사도 계산 및 유니티 전송 --------------#
    if isKeyPointTime():
        scores = calculatePoseSimilarities(new_landmarks1, new_landmarks2)
        print("유사도:", scores)
        multi_person_landmarks_3D = [
            [(lm.x, lm.y, lm.z) for lm in landmarks]
            for landmarks in pose_landmarker_result.pose_landmarks
        ]

        send_pose_to_unity(scores)
        send_pose_to_unity(multi_person_landmarks_3D)

    # -------------- 시각화 --------------#
    frame = drawLandmarksOnImage(frame, pose_landmarker_result)

    success, answer_pose_canvas = draw_answer_pose_on_canvas(
        new_landmarks1, new_landmarks2
    )

    if success:
        cv2.imshow("Answer Pose", answer_pose_canvas)
    cv2.imshow("Dance Pose Estimation", frame)

    # 'q' 키를 누르면 종료
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# 카메라와 윈도우 해제
cap.release()
cv2.destroyAllWindows()
