import cv2, platform, time, threading
from collections import deque
import socket  # UDP 리스너에 필요 (원래 코드 내에서 누락되어 있어 추가)
import numpy as np
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
        cap = cv2.VideoCapture(1)

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1080)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1920)
    cap.set(cv2.CAP_PROP_FPS, 60)

    if not cap.isOpened():
        print("* 카메라를 열 수 없습니다.")
        exit(0)

    print("* 카메라 준비 완료.")
    return cap


def udp_listener():
    """
    UDP 메시지를 수신하고, "start 2" 메시지를 받으면 is_dancing_time을 True로 설정.
    """
    global is_dancing_time, song_id
    udp_ip = "127.0.0.1"  # Unity가 메시지를 보내는 IP 주소 (localhost)
    udp_port = 25252  # Unity가 메시지를 보내는 포트

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind((udp_ip, udp_port))

    print("* UDP 메시지 수신 대기 중...")

    while True:
        try:
            data, addr = sock.recvfrom(1024)  # 최대 1024 바이트 메시지 수신
            message = data.decode("utf-8")
            print(f"* Unity 메시지 수신: {message}")

            # "start <song_id>" 메시지를 확인
            if message.startswith("start"):
                print(f"* 곡 {song_id} 시작 신호를 받았습니다")
                print("* 3.....")
                time.sleep(1)
                print("* 2.....")
                time.sleep(1)
                print("* 1.....")
                time.sleep(1)
                print("* 춤 시작!")
                _, song_id_str = message.split()
                song_id = int(song_id_str)  # song_id 업데이트
                is_dancing_time = True

        except Exception as e:
            print("* UDP 수신 중 에러 발생:", e)
            break


# 프레임 큐와 스레드 컨트롤 플래그 추가
frame_queue = deque(maxlen=2)
stop_flag = False


def frame_reader(cap):
    """카메라에서 프레임을 읽어오는 전용 스레드 함수"""
    global stop_flag
    while not stop_flag:
        ret, frame = cap.read()
        if not ret:
            time.sleep(0.001)
            continue
        # 프레임 전처리(반전)
        frame = cv2.flip(frame, 1)
        if len(frame_queue) == frame_queue.maxlen:
            frame_queue.popleft()
        frame_queue.append(frame)
        # CPU 점유율 감소를 위해 약간 대기
        time.sleep(0.001)


cap = initialize_camera()
database = load_dance_database()

udp_thread = threading.Thread(target=udp_listener, daemon=True)
udp_thread.start()

# 프레임 읽기용 스레드 시작
frame_thread = threading.Thread(target=frame_reader, args=(cap,), daemon=True)
frame_thread.start()

cv2.namedWindow("Dance Pose Estimation", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Dance Pose Estimation", 1920, 1080)


print("===========" * 10)
print("* 포즈 추정 시작")
print("* Unity의 곡 선정을 기다리고 있습니다...")

start_time = time.time()
is_dancing_time = False
song_id = 0

try:
    while True:
        # 큐에서 최신 프레임 가져오기
        if len(frame_queue) == 0:
            # 큐에 프레임이 없다면 잠시 대기 후 다음 루프
            time.sleep(0.005)
            continue

        frame = frame_queue[-1]
        current_time = round(time.time() - start_time, 1)

        (
            num_of_humans,
            sorted_real_time_2Ds,
            sorted_centered_real_time_2Ds,
            sorted_real_time_3Ds,
        ) = process_frame(frame)

        canvas = np.ones((500, 500, 3), dtype=np.uint8) * 255
        if not num_of_humans > 0:
            cv2.imshow("Dance Pose Estimation", frame)
            cv2.imshow("Answer Pose Comparisons", canvas)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
            if not is_dancing_time:
                start_time = time.time()
            continue

        try:
            width, height, centered_answer_2D, answer_3D = read_answer(
                database, song_id, current_time, is_dancing_time
            )
        except Exception:
            # 곡 종료 시 is_dancing_time False
            is_dancing_time = False
            # print("* 곡이 종료되었습니다.")  # 필요시 주석 유지

        unity_scores_and_poses = {0: (100.0, answer_3D)}

        similarity_scores = calculate_pose_similarity_vectorized(
            num_of_humans, centered_answer_2D, sorted_centered_real_time_2Ds
        )

        for human_index, similarity_score in enumerate(similarity_scores):
            real_time_2D = sorted_real_time_2Ds[human_index]
            real_time_3D = sorted_real_time_3Ds[human_index]

            unity_scores_and_poses[human_index + 1] = (
                similarity_score,
                real_time_3D.tolist(),
            )

            # 시각화 부분
            frame = painter.draw_realtime_frame(frame, human_index + 1, real_time_2D)
            canvas = painter.draw_pose_comparisons(
                canvas,
                human_index + 1,
                centered_answer_2D,
                sorted_centered_real_time_2Ds[human_index],
            )

        # 필요시 디버깅용 출력 최소화
        if is_dancing_time:
            print("개인별 점수:", similarity_scores)

        if is_keypoint_time(current_time):
            send_to_unity(unity_scores_and_poses)

        cv2.imshow("Dance Pose Estimation", frame)
        cv2.imshow("Answer Pose Comparisons", canvas)

        if not is_dancing_time:
            start_time = time.time()

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

except Exception as e:
    print("* 에러 발생:", e)

finally:
    print("* 자원을 반납합니다.")
    stop_flag = True
    cap.release()
    cv2.destroyAllWindows()
    print("* 정상적으로 종료되었습니다.")
