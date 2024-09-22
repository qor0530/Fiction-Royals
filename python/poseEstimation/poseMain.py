import cv2
from poseFunctions import *
from unityCommunication import *
from dataHandler import *


# 영상 및 정답 DB 로딩
key_points = loadDanceDatabase()

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

    # -------------- 유사도 계산 및 유니티 전송 --------------#
    if isKeyPointTime():
        scores = calculatePoseSimilarities(key_points, pose_landmarker_result)
        multi_person_landmarks_3D = [
            [(lm.x, lm.y, lm.z) for lm in landmarks]
            for landmarks in pose_landmarker_result.pose_landmarks
        ]

        send_pose_to_unity(scores)
        send_pose_to_unity(multi_person_landmarks_3D)

    # -------------- 시각화 --------------#
    frame = drawLandmarksOnImage(frame, pose_landmarker_result)
    cv2.imshow("Dance Pose Estimation", frame)

    # 'q' 키를 누르면 종료
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# 카메라와 윈도우 해제
cap.release()
cv2.destroyAllWindows()
