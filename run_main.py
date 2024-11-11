import cv2, platform, time
from python.poseEstimation.modules.poseFunctions import *
from python.poseEstimation.modules.unityCommunication import *
from python.poseEstimation.modules.dataHandler import *
from python.poseEstimation.modules.poseDrawing import *


def initialize_camera():
    print("* 카메라 준비 중...")
    cap = None
    if platform.system() == "Windows":
        cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    else:
        cap = cv2.VideoCapture(0)

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
    cap.set(cv2.CAP_PROP_FPS, 60)

    if not cap.isOpened():
        print("* 카메라를 열 수 없습니다.")
        exit(0)

    print("* 카메라 준비 완료.")
    return cap


cap = initialize_camera()
database = load_dance_database()

cv2.namedWindow("Dance Pose Estimation", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Dance Pose Estimation", 640, 360)


print("===========" * 10)
print("* 포즈 추정 시작")

start_time = time.time()

try:
    while True:
        ret, frame = cap.read()
        frame = cv2.flip(frame, 1)
        canvas = np.ones((500, 500, 3), dtype=np.uint8) * 255
        current_time = round(time.time() - start_time, 1)
        if not ret:
            break

        num_of_humans, landmarks_2d, landmarks_2d_normed, landmarks_3d = process_frame(
            frame
        )

        if landmarks_2d_normed is not None:
            normalized_answer_2d, answer_landmarks_3d = read_answer(
                database, current_time
            )
            unity_scores_and_poses = {0: (100.0, answer_landmarks_3d)}

            similarity_scores = calculate_pose_similarity_vectorized(
                np.array([normalized_answer_2d] * len(landmarks_2d_normed)),
                np.array(landmarks_2d_normed),
            )

            for human_index, similarity_score in enumerate(similarity_scores):
                landmark_2d = landmarks_2d[human_index]
                normalized_real_time_2d = landmarks_2d_normed[human_index]
                real_time_landmarks_3d = landmarks_3d[human_index]

                # 그림 그리기
                frame = painter.draw_realtime_frame(frame, human_index, landmark_2d)
                canvas = painter.draw_pose_comparisons(
                    canvas,
                    human_index,
                    normalized_answer_2d,
                    normalized_real_time_2d,
                )

                # 유니티 데이터 추가
                unity_scores_and_poses[human_index + 1] = (
                    similarity_score,
                    real_time_landmarks_3d.tolist(),
                )

            print("Scores:", similarity_scores)
            if is_keypoint_time(current_time):
                send_to_unity(unity_scores_and_poses)
                pass

        cv2.imshow("Dance Pose Estimation", frame)
        cv2.imshow("Answer Pose Comparisons", canvas)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

except Exception as e:
    print("* 에러 발생:", e)

finally:
    print("* 자원을 반납합니다.")
    cap.release()
    cv2.destroyAllWindows()
    print("* 정상적으로 종료되었습니다.")
