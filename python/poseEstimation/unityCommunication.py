import socket
import json


def send_pose_to_unity(data):
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        unity_address = (
            "117.16.153.148",
            25712,
        )

        # print(data)
        message = json.dumps(data)
        sock.sendto(message.encode("utf-8"), unity_address)

        sock.close()
        # print("성공")
        return True
    except Exception as e:
        print(f"Network error: {e}")
        return False


# if __name__ == "__main__":
#     import socket

#     # 서버 설정
#     UDP_IP = "127.0.0.1"
#     UDP_PORT = 5005

#     # 소켓 생성
#     sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
#     sock.bind((UDP_IP, UDP_PORT))

#     print(f"UDP 서버가 {UDP_IP}:{UDP_PORT} 에서 리스닝 중입니다...")

#     # 데이터 수신
#     while True:
#         data, addr = sock.recvfrom(1024)  # 버퍼 크기 1024
#         print(f"받은 데이터: {data.decode('utf-8')} 주소: {addr}")
