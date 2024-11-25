import cv2
import mediapipe as mp
import numpy as np
import os

mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils


def extract_pose(video_path, output_video_path):
    if not os.path.isfile(video_path):
        print(f"비디오 파일 {video_path}을 찾을 수 없습니다.")
        return [], None, None

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"비디오 스트림 {video_path}을 제대로 읽을 수 없습니다.")
        return [], None, None

    # 고정할 FPS 설정
    fixed_fps = 10

    # 비디오 정보 가져오기
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    original_fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    video_length = total_frames / original_fps

    # 비디오 저장 설정
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(
        output_video_path, fourcc, fixed_fps, (frame_width, frame_height)
    )

    pose = mp_pose.Pose(
        static_image_mode=False,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )

    # 결과 저장용 리스트
    pose_3d_results = []
    pose_2d_results = []

    # 프레임 간격 설정
    frame_interval = original_fps / fixed_fps

    frame_count = 0
    accumulated_frame = 0.0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # 설정한 프레임 간격에 맞춰 프레임을 샘플링
        if round(accumulated_frame) == frame_count:
            # BGR을 RGB로 변환
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image.flags.writeable = False

            # 포즈 추출
            results = pose.process(image)

            # 포즈가 추출되었으면
            pose_3d, pose_2d = [], []
            if results.pose_landmarks:
                for landmark_2d, landmark_3d in zip(
                    results.pose_landmarks.landmark,
                    results.pose_world_landmarks.landmark,
                ):
                    pose_2d.append((landmark_2d.x, landmark_2d.y))
                    pose_3d.append((landmark_3d.x, landmark_3d.y, landmark_3d.z))
            else:
                # 포즈가 아예 인식되지 않으면 모든 관절을 None으로 처리
                for _ in range(33):  # 포즈 랜드마크는 총 33개
                    pose_2d.append((None, None))
                    pose_3d.append((None, None, None))

            pose_2d_results.append(pose_2d)
            pose_3d_results.append(pose_3d)

            # 포즈 시각화
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            mp_drawing.draw_landmarks(
                image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS
            )
            cv2.imshow("3D Pose", image)
            out.write(image)
            accumulated_frame += frame_interval

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        frame_count += 1

    cap.release()
    out.release()
    cv2.destroyAllWindows()

    return (
        pose_2d_results,
        pose_3d_results,
        (frame_width, frame_height),
        (1000 / fixed_fps),
        video_length,
    )


def save_pose_data_as_txt(
    pose_2ds, pose_3ds, output_file, frame_size, frame_interval, video_length
):
    with open(output_file, "w") as f:
        # 헤더 정보 저장
        f.write(f"{frame_size[0]},{frame_size[1]},{frame_interval},{video_length}\n")

        # 포즈 데이터 저장
        for p_2d, p_3d in zip(pose_2ds, pose_3ds):
            for (x_2d, y_2d), (x_3d, y_3d, z_3d) in zip(p_2d, p_3d):
                f.write(f"{x_2d},{y_2d},{x_3d},{y_3d},{z_3d}\n")
            f.write("\n")


def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    video_dir = os.path.join(script_dir, "..", "..", "db", "videoFile")
    output_dir = os.path.join(script_dir, "..", "..", "db", "3")  # 파일 위치

    video_filename = "supernova.mp4"  # 비디오 파일 이름적기

    video_path = os.path.join(video_dir, video_filename)
    output_txt_file = os.path.join(output_dir, f"poses.txt")
    output_video_path = os.path.join(output_dir, f"output_video_{video_filename}")

    print(video_path, output_txt_file, output_video_path, sep="\n\n\n")

    if not os.path.isfile(video_path):
        print(f"{video_path} 파일이 존재하지 않습니다.")
        return

    # 관절 데이터 추출
    pose_2ds, pose_3ds, frame_size, frame_interval, video_length = extract_pose(
        video_path, output_video_path
    )

    save_pose_data_as_txt(
        pose_2ds, pose_3ds, output_txt_file, frame_size, frame_interval, video_length
    )
    print(f"포즈 데이터를 {output_txt_file}에 저장했습니다.")
    print(f"포즈 비디오를 {output_video_path}에 저장했습니다.")


if __name__ == "__main__":
    main()
