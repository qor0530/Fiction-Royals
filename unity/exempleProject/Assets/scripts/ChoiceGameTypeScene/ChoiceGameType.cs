using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.SceneManagement;
using UnityEngine.UI; // UI 버튼을 사용하기 위해 필요

public class ChoiceGameType : MonoBehaviour
{
    // 버튼들을 인스펙터에서 연결할 수 있게 설정
    public Button singleBtn;
    public Button multiBtn;

    void Start()
    {
        // 싱글플레이 버튼 클릭 시 싱글 모드 선택
        singleBtn.onClick.AddListener(() => SetGameMode(true));

        // 멀티플레이 버튼 클릭 시 멀티 모드 선택
        multiBtn.onClick.AddListener(() => SetGameMode(false));
    }

    // 게임 모드 설정 및 씬 전환
    void SetGameMode(bool isSinglePlayer)
    {
        // 싱글 모드면 1, 멀티 모드면 2 저장
        PlayerPrefs.SetInt("GameMode", isSinglePlayer ? 1 : 2);
        PlayerPrefs.Save();

        // GameScene으로 씬 전환
        SceneManager.LoadScene("SelectScene");
    }
}
