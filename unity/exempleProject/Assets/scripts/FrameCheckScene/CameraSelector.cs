using UnityEngine;
using UnityEngine.UI;
using TMPro;
using UnityEngine.SceneManagement;

public class CameraSelector : MonoBehaviour
{
    public RawImage display;  // 캠 화면을 출력할 UI RawImage
    public TMP_Dropdown cameraDropdown;  // TextMeshPro Dropdown UI
    public Button nextSceneButton;  // 다음 씬으로 이동할 버튼

    private WebCamTexture webcamTexture;
    private WebCamDevice[] availableCameras;  // 사용 가능한 카메라 목록

    void Start()
    {
        // 사용 가능한 카메라 장치를 가져옴
        availableCameras = WebCamTexture.devices;

        // 카메라 목록을 Dropdown에 추가
        cameraDropdown.ClearOptions();
        foreach (var device in availableCameras)
        {
            cameraDropdown.options.Add(new TMP_Dropdown.OptionData(device.name));
        }

        // 기본적으로 첫 번째 카메라를 선택하고 출력
        cameraDropdown.onValueChanged.AddListener(delegate { SelectCamera(cameraDropdown.value); });
        SelectCamera(0);

        // 버튼 클릭 시 다음 씬으로 이동
        nextSceneButton.onClick.AddListener(GoToGameScene);
    }

    // 사용자가 선택한 카메라로 변경하는 함수
    void SelectCamera(int index)
    {
        if (webcamTexture != null)
        {
            webcamTexture.Stop();
        }

        // 선택된 카메라로 WebCamTexture 생성
        webcamTexture = new WebCamTexture(availableCameras[index].name);
        display.texture = webcamTexture;
        display.material.mainTexture = webcamTexture;

        // 웹캠 시작
        webcamTexture.Play();

        // 선택된 카메라 인덱스를 PlayerPrefs에 저장
        PlayerPrefs.SetInt("SelectedCameraIndex", index);
        PlayerPrefs.Save();
    }

    // 버튼 클릭 시 다음 씬(GameScene)으로 이동하는 함수
    public void GoToGameScene()
    {
        SceneManager.LoadScene("GameScene");
    }

    void OnDisable()
    {
        // 웹캠을 중지함
        if (webcamTexture != null)
        {
            webcamTexture.Stop();
        }
    }
}
