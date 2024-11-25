import socket


def send_udp_message(message, ip="127.0.0.1", port=25252):
    """
    UDP 메시지를 전송하는 함수
    :param message: 전송할 메시지 (문자열)
    :param ip: 수신자 IP 주소 (기본값: localhost)
    :param port: 수신자 포트 번호 (기본값: 12345)
    """
    try:
        # 소켓 생성
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

        # 메시지 전송
        sock.sendto(message.encode("utf-8"), (ip, port))
        print(f"* 메시지 '{message}'를 {ip}:{port}로 전송했습니다.")

        # 소켓 닫기
        sock.close()
    except Exception as e:
        print(f"* 메시지 전송 실패: {e}")


# 사용 예시
if __name__ == "__main__":
    song_id = input("몇번 곡을 실행할까요? >> ")
    send_udp_message(f"start {song_id}")  # "start 3" 메시지 전송
