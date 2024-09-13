using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class GameModeChecker : MonoBehaviour
{
    void Start()
    {
        // PlayerPrefs에서 게임 모드 값 불러오기 (기본값은 싱글플레이 1)
        int gameMode = PlayerPrefs.GetInt("GameMode", 1);

        if (gameMode == 1)
        {
            Debug.Log("싱글플레이 모드 시작");
        }
        else
        {
            Debug.Log("멀티플레이 모드 시작");
        }
    }
}
