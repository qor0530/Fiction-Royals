import cv2
from poseFunctions import *
from unityCommunication import *
from dataHandler import *
from poseDrawing import *

# 정답 포즈 이미지와 카메라 초기 설정

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("카메라를 열 수 없습니다.")
    exit(0)

key_points = loadDanceDatabase()
answer_image = cv2.imread("python/poseEstimation/dance.jpg")
answer_pose_landmarks = landmarker.detect(
    mp.Image(
        image_format=mp.ImageFormat.SRGBA,
        data=cv2.cvtColor(answer_image, cv2.COLOR_BGR2RGBA),
    )
)

print("=" * 70)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    landmarks_pairs = process_frame(frame, answer_pose_landmarks.pose_landmarks)

    if landmarks_pairs is None:
        cv2.imshow("Dance Pose Estimation", frame)
    else:
        for landmarks_answer, landmarks_real_time in landmarks_pairs:
            if isKeyPointTime():
                similarity_score = calculate_pose_similarity(
                    landmarks_answer, landmarks_real_time
                )
                print("유사도:", similarity_score)

                # 수정 필요
                # pose_landmarks_3D = [
                #     [(lm.x, lm.y, lm.z) for lm in landmarks]
                #     for landmarks in landmarks_answer.pose_landmarks
                # ]
                send_pose_to_unity(similarity_score)
                # send_pose_to_unity(pose_landmarks_3D)

            # 시각화 및 화면 표시
            frame = draw_landmarks_on_image(frame, landmarks_answer)
            success, answer_canvas = draw_answer_pose_on_canvas(
                landmarks_answer, landmarks_real_time
            )

            if success:
                cv2.imshow("Answer Pose", answer_canvas)
            cv2.imshow("Dance Pose Estimation", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
