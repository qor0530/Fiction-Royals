using System;
using System.Net;
using System.Net.Sockets;
using System.Text;
using System.Threading;
using UnityEngine;
using Polyperfect.People; // PolyIk Ŭ������ ���� ���ӽ����̽� �߰�

// PoseData Ŭ���� ���� �߰�
[Serializable]
public class PoseData
{
    public float x; // JSON �����Ϳ� �°� �ʵ� �߰�
    public float y; // JSON �����Ϳ� �°� �ʵ� �߰�
    public float z; // JSON �����Ϳ� �°� �ʵ� �߰�
}

public class UDPReceiver : MonoBehaviour
{
    private UdpClient udpClient;
    private Thread receiveThread;
    private int port = 25712; // ��Ʈ ��ȣ�� 25712�� ����
    private PolyIk polyIk;

    void Start()
    {
        // PolyIk ��ũ��Ʈ ã��
        polyIk = FindObjectOfType<PolyIk>();
        if (polyIk == null)
        {
            Debug.LogError("No PolyIk script found in the scene.");
            return;
        }

        // UDP Ŭ���̾�Ʈ �ʱ�ȭ
        udpClient = new UdpClient(port);

        // ���� ������ ����
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

                // ���� �ð� ���
                DateTime receiveTime = DateTime.Now;
                Debug.Log($"[UDP Received at {receiveTime:HH:mm:ss.fff}] Raw JSON: {message}");

                // JSON �Ľ�
                try
                {
                    PoseData poseData = JsonUtility.FromJson<PoseData>(message);
                    Debug.Log($"Parsed JSON - x: {poseData.x}, y: {poseData.y}, z: {poseData.z} (Received at {receiveTime:HH:mm:ss.fff})");

                    // ���ŵ� �����ͷ� Ÿ�� ��ġ ������Ʈ
                    Vector3 newPosition = new Vector3(poseData.x, poseData.y, poseData.z);
                    polyIk.SetTargetPosition(newPosition, "leftHand"); // "leftHand" �κ��� �װ� �̵���Ű�� ���� ������ �°� ����
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
