using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.SceneManagement;
using UnityEngine.UI;

public class ResultManager : MonoBehaviour
{
    
    public Button confirmResultButton; //확인버튼 : 누르면 랭킹화면으로 넘어감

    void Start()
    {
        PrintResult();
        // 확인버튼 클릭 이벤트 연결
        confirmResultButton.onClick.AddListener(ConfirmResult);
    }

    // 게임 결과 출력 함수
    void PrintResult()
    {
        //구조 : 싱 or 멀티 받고, 싱이면 result 하나 멀티면 result 여러개 출력
        Debug.Log("확인");
    }

    //result 확인 후 RankingScene으로 이동하는 함수
    public void ConfirmResult()
    {
        //RankingScene으로 전환
        SceneManager.LoadScene("RankingScene");
    }
}
