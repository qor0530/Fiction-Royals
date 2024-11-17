import socket
import json

TARGET_IP = "127.0.0.1"


def send_to_unity(data):
    """유니티 서버로 데이터를 전송하는 함수"""
    try:
        # JSON 호환 형식으로 변환
        data_json_compatible = {
            "data": [
                {"score": float(score), "coord_3d": real_time_landmarks_3d}
                for key, (score, real_time_landmarks_3d) in data.items()
            ]
        }

        # 소켓 설정
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        unity_address = (TARGET_IP, 25712)

        # JSON 형식으로 변환하여 전송
        message = json.dumps(data_json_compatible)
        sock.sendto(message.encode("utf-8"), unity_address)
        sock.close()
        return True

    except socket.error as e:
        print(f"Network error while sending data to Unity: {e}")
        return False
    except TypeError as e:
        print(f"Serialization error: {e}")
        return False


if __name__ == "__main__":
    import socket

    # 서버 설정
    UDP_IP = "127.0.0.1"
    UDP_PORT = 25712

    # 소켓 생성
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind((UDP_IP, UDP_PORT))

    print(f"UDP 서버가 {UDP_IP}:{UDP_PORT} 에서 리스닝 중입니다...")

    # 데이터 수신
    while True:
        data, addr = sock.recvfrom(16384)
        data = json.loads(data.decode("utf-8"))
        data = [info_dict for info_dict in data.values()]

        answer_3D_pose = data[0]["coord_3d"]
        print("--------------------------------------------")
        for user_id, d in enumerate(data[1:]):
            score, coord_3d = d["score"], d["coord_3d"]
            print(
                f"유저 {user_id+1}: {score} / 수신 좌표 형태: {len(coord_3d), len(coord_3d[0])}"
            )
        print()
