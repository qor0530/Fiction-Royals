import cv2, platform, time
from python.poseEstimation.modules.poseFunctions import *
from python.poseEstimation.modules.unityCommunication import *
from python.poseEstimation.modules.dataHandler import *
from python.poseEstimation.modules.poseDrawing import *


print("* 카메라 준비 중...")
cap = None
if platform.system() == "Windows":
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
else:
    cap = cv2.VideoCapture(0)

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
cap.set(cv2.CAP_PROP_FPS, 30)

if not cap.isOpened():
    print("* 카메라를 열 수 없습니다.")
    exit(0)
print("* 카메라 준비 완료.")


database = loadDanceDatabase()
print("* 데이터베이스 준비 완료.")

painter = Painter()
cv2.namedWindow("Dance Pose Estimation", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Dance Pose Estimation", 640, 360)
print("* 시각화 도구 준비 완료.")

print("===========" * 10)
print("* 포즈 추정 시작")
print("** Debug ** 유저는 왼쪽부터 1번입니다.(정답 모델은 0번)")

start_time = time.time()
try:
    while True:
        ret, frame = cap.read()
        canvas = np.ones((500, 500, 3), dtype=np.uint8) * 255
        current_time = round(time.time() - start_time, 1)
        if not ret:
            break

        ### n * 2 * (33 * (2 or 3)) 형태.
        ### n은 사람 수
        ### 2는 2D-3D 쌍 튜플(2D, 3D)
        ### 33은 사람 한명 관절 수.
        ### 2 or 3은 픽셀 or 3D 좌표
        landmarks_realtime = process_frame(frame)
        normalized_answer_2d, answer_landmarks_3d = read_answer(database, current_time)
        unity_scores_and_poses = [(1.0, answer_landmarks_3d)]

        for human_index, (normalized_real_time_2d, real_time_landmarks_3d) in enumerate(
            landmarks_realtime
        ):

            # 유사도 분석
            similarity_score = calculate_pose_similarity(
                normalized_answer_2d,
                normalized_real_time_2d,
            )
            print(f"{human_index + 1}번 유저의 포즈 유사도:", similarity_score)

            # 그림 그리기
            frame = painter.draw_realtime_frame(
                frame, human_index, normalized_real_time_2d
            )
            canvas = painter.draw_pose_comparisons(
                canvas,
                human_index,
                normalized_answer_2d,
                normalized_real_time_2d,
            )

            # 유니티 데이터 추가
            unity_scores_and_poses.append((similarity_score, real_time_landmarks_3d))

        if isKeyPointTime(current_time):
            send_to_unity(unity_scores_and_poses)

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
