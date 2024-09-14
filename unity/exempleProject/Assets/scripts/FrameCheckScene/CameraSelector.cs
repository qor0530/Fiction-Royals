using UnityEngine;
using UnityEngine.UI;
using TMPro;
using UnityEngine.SceneManagement;

public class CameraSelector : MonoBehaviour
{
    public RawImage display;  // ķ ȭ���� ����� UI RawImage
    public TMP_Dropdown cameraDropdown;  // TextMeshPro Dropdown UI
    public Button nextSceneButton;  // ���� ������ �̵��� ��ư

    private WebCamTexture webcamTexture;
    private WebCamDevice[] availableCameras;  // ��� ������ ī�޶� ���

    void Start()
    {
        // ��� ������ ī�޶� ��ġ�� ������
        availableCameras = WebCamTexture.devices;

        // ī�޶� ����� Dropdown�� �߰�
        cameraDropdown.ClearOptions();
        foreach (var device in availableCameras)
        {
            cameraDropdown.options.Add(new TMP_Dropdown.OptionData(device.name));
        }

        // �⺻������ ù ��° ī�޶� �����ϰ� ���
        cameraDropdown.onValueChanged.AddListener(delegate { SelectCamera(cameraDropdown.value); });
        SelectCamera(0);

        // ��ư Ŭ�� �� ���� ������ �̵�
        nextSceneButton.onClick.AddListener(GoToGameScene);
    }

    // ����ڰ� ������ ī�޶�� �����ϴ� �Լ�
    void SelectCamera(int index)
    {
        if (webcamTexture != null)
        {
            webcamTexture.Stop();
        }

        // ���õ� ī�޶�� WebCamTexture ����
        webcamTexture = new WebCamTexture(availableCameras[index].name);
        display.texture = webcamTexture;
        display.material.mainTexture = webcamTexture;

        // ��ķ ����
        webcamTexture.Play();

        // ���õ� ī�޶� �ε����� PlayerPrefs�� ����
        PlayerPrefs.SetInt("SelectedCameraIndex", index);
        PlayerPrefs.Save();
    }

    // ��ư Ŭ�� �� ���� ��(GameScene)���� �̵��ϴ� �Լ�
    public void GoToGameScene()
    {
        SceneManager.LoadScene("GameScene");
    }

    void OnDisable()
    {
        // ��ķ�� ������
        if (webcamTexture != null)
        {
            webcamTexture.Stop();
        }
    }
}
