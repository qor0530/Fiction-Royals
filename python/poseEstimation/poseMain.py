import cv2
from poseFunctions import *
from unityCommunication import *

# 변수 초기화
cameraError = False
poseError = False
networkError = False

# 영상 및 정답 DB 로딩
keyPoints = []
myPoints = []

while True:
    # -------------- 카메라 처리 --------------#
    pass

    # -------------- 포즈 처리 --------------#
    # 1. YOLO Human Bounding Box 처리
    pass
    # 2. 각 Bounding Box에 대한 Mediapipe Pose Estimation 진행
    pass

    # -------------- 유사도 계산(시은) -------------- #
    pass

    # -------------- unity 전송 --------------#
    pass

    if cameraError or poseError or networkError:
        break
