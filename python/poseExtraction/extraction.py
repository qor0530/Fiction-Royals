import cv2
import mediapipe as mp
import numpy as np
import os

mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils


def extract_3d_pose(video_path, output_video_path):
    if not os.path.isfile(video_path):
        print(f"비디오 파일 {video_path}을 찾을 수 없습니다.")
        return [], None, None

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"비디오 스트림 {video_path}을 제대로 읽을 수 없습니다.")
        return [], None, None

    # 비디오 정보 가져오기
    fps = cap.get(cv2.CAP_PROP_FPS)  # 30으로 나옴
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    video_length = total_frames / fps

    # 비디오 저장 설정
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video_path, fourcc, fps,
                          (frame_width, frame_height))

    pose = mp_pose.Pose(static_image_mode=False,
                        min_detection_confidence=0.5, min_tracking_confidence=0.5)

    # 결과 저장용 리스트
    pose_3d_results = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # BGR을 RGB로 변환
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False

        # 포즈 추출
        results = pose.process(image)

        # 포즈가 추출되었으면
        if results.pose_landmarks:
            pose_3d = []
            for landmark in results.pose_landmarks.landmark:
                pose_3d.append({
                    "x": landmark.x,  # 정규화된 x 좌표
                    "y": landmark.y,  # 정규화된 y 좌표
                })
            pose_3d_results.append(pose_3d)

            # 포즈 시각화
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            mp_drawing.draw_landmarks(
                image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
            cv2.imshow('3D Pose', image)
            out.write(image)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    cap.release()
    out.release()
    cv2.destroyAllWindows()
    return pose_3d_results, (frame_width, frame_height), (1000 / fps), video_length


def save_pose_data_as_txt(pose_data, output_file, frame_size, frame_interval, video_length):
    with open(output_file, 'w') as f:
        # 헤더 정보 저장
        f.write(
            f"{frame_size[0]},{frame_size[1]},{frame_interval},{video_length}\n")

        # 포즈 데이터 저장
        for frame_data in pose_data:
            for point in frame_data:
                f.write(f"{point['x']},{point['y']}\n")
            f.write("\n")


def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    video_dir = os.path.join(script_dir, 'videoFile')
    output_dir = os.path.join(script_dir, 'outputData')

    video_filename = 'tell_me_short.mp4'  # 비디오 파일 이름
    video_path = os.path.join(video_dir, video_filename)

    output_file = os.path.join(output_dir, 'output_pose_data.txt')
    output_video_path = os.path.join(output_dir, 'output_pose_video.mp4')

    pose_data, frame_size, frame_interval, video_length = extract_3d_pose(
        video_path, output_video_path)
    if pose_data:
        save_pose_data_as_txt(pose_data, output_file,
                              frame_size, frame_interval, video_length)
        print(f"포즈 데이터를 {output_file}에 저장했습니다.")
        print(f"포즈 비디오를 {output_video_path}에 저장했습니다.")
    else:
        print("저장할 포즈 데이터가 없습니다.")


if __name__ == "__main__":
    main()
