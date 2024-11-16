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

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
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

        (
            num_of_humans,
            sorted_real_time_2Ds,
            sorted_centered_real_time_2Ds,
            sorted_real_time_3Ds,
        ) = process_frame(frame)

        if num_of_humans > 0:
            width, height, centered_answer_2D, answer_3D = read_answer(
                database, current_time
            )
            unity_scores_and_poses = {0: (100.0, answer_3D)}

            similarity_scores = calculate_pose_similarity_vectorized(
                num_of_humans, centered_answer_2D, sorted_centered_real_time_2Ds
            )

            for human_index, similarity_score in enumerate(similarity_scores):
                real_time_2D = sorted_real_time_2Ds[human_index]
                real_time_3D = sorted_real_time_3Ds[human_index]
                centered_real_time_2D = sorted_centered_real_time_2Ds[human_index]

                # 유니티 데이터 추가
                unity_scores_and_poses[human_index + 1] = (
                    similarity_score,
                    real_time_3D.tolist(),
                )

                # 그림 그리기
                frame = painter.draw_realtime_frame(
                    frame, human_index + 1, real_time_2D
                )
                canvas = painter.draw_pose_comparisons(
                    canvas,
                    human_index + 1,
                    centered_answer_2D,
                    centered_real_time_2D,
                )

            print("개인별 점수:", similarity_scores)
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
