using UnityEngine;
using UnityEngine.SceneManagement;
using UnityEngine.UI;
using System.Collections.Generic;

public class MusicSelector : MonoBehaviour
{
    // 썸네일 이미지 리스트 (유니티 에디터에서 추가 가능)
    public List<Sprite> thumbnailList;
    
    // 썸네일을 표시할 UI Image
    public Image displayImage;
    
    // 좌우 화살표 버튼
    public Button leftArrowButton;
    public Button rightArrowButton;

    // 현재 선택된 썸네일 인덱스
    private int currentIndex = 0;

    // 선택된 썸네일을 클릭하면 씬 이동을 할 수 있도록 Button으로 참조
    public Button thumbnailButton;

    void Start()
    {
        // 좌우 화살표 버튼에 이벤트 연결
        leftArrowButton.onClick.AddListener(() => ChangeThumbnail(-1));
        rightArrowButton.onClick.AddListener(() => ChangeThumbnail(1));

        // 썸네일 클릭 시 음악을 선택하고 씬 전환
        thumbnailButton.onClick.AddListener(SelectMusic);

        // 첫 번째 썸네일을 보여줌
        UpdateThumbnail();
    }

    // 썸네일을 변경하는 함수
    void ChangeThumbnail(int direction)
    {
        // 인덱스 업데이트 (리스트의 크기를 넘지 않도록 처리)
        currentIndex += direction;
        if (currentIndex < 0)
        {
            currentIndex = thumbnailList.Count - 1;
        }
        else if (currentIndex >= thumbnailList.Count)
        {
            currentIndex = 0;
        }

        // 썸네일을 업데이트
        UpdateThumbnail();
    }

    // 현재 인덱스에 맞는 썸네일을 화면에 표시
    void UpdateThumbnail()
    {
        if (thumbnailList.Count > 0)
        {
            displayImage.sprite = thumbnailList[currentIndex];
        }
    }

    // 선택된 음악을 저장하고 게임을 시작할 수 있는 함수
    public void SelectMusic()
    {
        PlayerPrefs.SetInt("SelectedMusicIndex", currentIndex); // 선택된 음악 인덱스를 저장
        PlayerPrefs.Save();

        SceneManager.LoadScene("FrameCheckScene");
    }
}
