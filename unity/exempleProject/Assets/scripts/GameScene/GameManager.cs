using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.UI; // UI 관련 네임스페이스s
using UnityEngine.SceneManagement; // 씬 전환을 위한 네임스페이스

public class GameManager : MonoBehaviour
{

    public Button completeGameButton; // 게임 완료 버튼

    void Start()
    {
        // PlayerPrefs에 저장된 카메라 및 썸네일 정보를 로그로 출력
        PrintAllPlayerPrefs();
        // 게임 완료 버튼 클릭 이벤트 연결
        completeGameButton.onClick.AddListener(CompleteGame);
    }



    // PlayerPrefs에 저장된 모든 내용을 출력하는 함수 (중복 제거)
    void PrintAllPlayerPrefs()
    {
        // 저장된 GameMode 불러오기
        if (PlayerPrefs.HasKey("GameMode"))
        {
            int gameMode = PlayerPrefs.GetInt("GameMode");

            if (gameMode == 1)
            {
                Debug.Log("싱글플레이 모드로 시작");
            }
            else if (gameMode == 2)
            {
                Debug.Log("멀티플레이 모드로 시작");
            }
            else
            {
                Debug.Log("잘못된 gameMode입니다.");
            }
        }
        else
        {
            Debug.Log("GameMode를 찾을 수 없습니다.");
        }


        // 선택된 카메라 정보 출력
        if (PlayerPrefs.HasKey("SelectedCameraIndex"))
        {
                int selectedCameraIndex = PlayerPrefs.GetInt("SelectedCameraIndex");
                Debug.Log("SelectedCameraIndex: " + selectedCameraIndex);
        }
        else
        {
            Debug.Log("카메라 정보를 찾을 수 없습니다.");
        }

        // 선택된 썸네일 정보 출력
        if (PlayerPrefs.HasKey("SelectedMusicIndex"))
        {
            int selectedThumbnailIndex = PlayerPrefs.GetInt("SelectedMusicIndex");
            Debug.Log("SelectedMusicIndex: " + selectedThumbnailIndex);
        }
        else
        {
            Debug.Log("음악 정보를 찾을 수 없습니다.");
        }

        Debug.Log("PlayerPrefs 저장된 모든 정보를 출력했습니다.");
    }  
    // 게임 완료 시 ResultScene으로 이동하는 함수
    public void CompleteGame()
    {
        // ResultScene으로 전환
        SceneManager.LoadScene("ResultScene");
    }
}