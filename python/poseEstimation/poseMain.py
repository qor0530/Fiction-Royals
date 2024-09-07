import cv2, time
from poseFunctions import *
from unityCommunication import *
from dataHandler import *

# 초기 변수 설정
number_of_dancers = 2  # 댄서 수

# 영상 및 정답 DB 로딩
key_points = load_dance_database()

# 카메라 초기화
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("카메라를 여는데 실패했습니다.")
    exit(0)

width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)


while True:
    # -------------- 카메라 처리 --------------#
    ret, frame = cap.read()
    if not ret:
        break

    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)

    pose_landmarker_result = detector.detect(mp_image)

    annotated_image = draw_landmarks_on_image(
        mp_image.numpy_view(), pose_landmarker_result
    )

    # # -------------- Unity 전송 --------------#
    # if not send_pose_to_unity(scores):
    #     print("UDP 전달에 실패했습니다.")
    #     break

    print(pose_landmarker_result)

    cv2.imshow("Dance Pose Estimation", annotated_image)

    # 'q' 키를 누르면 종료
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# 카메라와 윈도우 해제
cap.release()
cv2.destroyAllWindows()
