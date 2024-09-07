# unityCommunication.py
import socket
import json


def send_pose_to_unity(pose_data):
    try:
        # Setup UDP connection to Unity
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        unity_address = (
            "127.0.0.1",
            5005,
        )  # Replace with your Unity server IP and port

        # Convert pose data to JSON
        message = json.dumps(pose_data)
        sock.sendto(message.encode("utf-8"), unity_address)

        sock.close()
        return True
    except Exception as e:
        print(f"Network error: {e}")
        return False
