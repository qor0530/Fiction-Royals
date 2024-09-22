using System;
using System.Net;
using System.Net.Sockets;
using System.Text;
using System.Threading;
using UnityEngine;
using Polyperfect.People; // PolyIk 클래스가 속한 네임스페이스 추가

// PoseData 클래스 정의 추가
[Serializable]
public class PoseData
{
    public float x; // JSON 데이터에 맞게 필드 추가
    public float y; // JSON 데이터에 맞게 필드 추가
    public float z; // JSON 데이터에 맞게 필드 추가
}

public class UDPReceiver : MonoBehaviour
{
    private UdpClient udpClient;
    private Thread receiveThread;
    private int port = 25712; // 포트 번호를 25712로 변경
    private PolyIk polyIk;

    void Start()
    {
        // PolyIk 스크립트 찾기
        polyIk = FindObjectOfType<PolyIk>();
        if (polyIk == null)
        {
            Debug.LogError("No PolyIk script found in the scene.");
            return;
        }

        // UDP 클라이언트 초기화
        udpClient = new UdpClient(port);

        // 수신 스레드 시작
        receiveThread = new Thread(new ThreadStart(ReceiveData));
        receiveThread.IsBackground = true;
        receiveThread.Start();

        Debug.Log($"Started UDP listener on port {port}");
    }

    void ReceiveData()
    {
        IPEndPoint remoteEndPoint = new IPEndPoint(IPAddress.Any, port);

        try
        {
            while (true)
            {
                byte[] data = udpClient.Receive(ref remoteEndPoint);
                string message = Encoding.UTF8.GetString(data);

                // 수신 시간 기록
                DateTime receiveTime = DateTime.Now;
                Debug.Log($"[UDP Received at {receiveTime:HH:mm:ss.fff}] Raw JSON: {message}");

                // JSON 파싱
                try
                {
                    PoseData poseData = JsonUtility.FromJson<PoseData>(message);
                    Debug.Log($"Parsed JSON - x: {poseData.x}, y: {poseData.y}, z: {poseData.z} (Received at {receiveTime:HH:mm:ss.fff})");

                    // 수신된 데이터로 타겟 위치 업데이트
                    Vector3 newPosition = new Vector3(poseData.x, poseData.y, poseData.z);
                    polyIk.SetTargetPosition(newPosition, "leftHand"); // "leftHand" 부분을 네가 이동시키고 싶은 부위에 맞게 변경
                }
                catch (Exception e)
                {
                    Debug.LogError($"Failed to parse JSON: {e.Message}");
                }
            }
        }
        catch (Exception e)
        {
            Debug.LogError($"Error receiving data: {e.Message}");
        }
    }

    void OnApplicationQuit()
    {
        if (receiveThread != null)
            receiveThread.Abort();

        udpClient.Close();
    }
}
