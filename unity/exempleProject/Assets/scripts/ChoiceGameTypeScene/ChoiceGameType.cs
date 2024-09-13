using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.SceneManagement;
using UnityEngine.UI; // UI ��ư�� ����ϱ� ���� �ʿ�

public class ChoiceGameType : MonoBehaviour
{
    // ��ư���� �ν����Ϳ��� ������ �� �ְ� ����
    public Button singleBtn;
    public Button multiBtn;

    void Start()
    {
        // �̱��÷��� ��ư Ŭ�� �� �̱� ��� ����
        singleBtn.onClick.AddListener(() => SetGameMode(true));

        // ��Ƽ�÷��� ��ư Ŭ�� �� ��Ƽ ��� ����
        multiBtn.onClick.AddListener(() => SetGameMode(false));
    }

    // ���� ��� ���� �� �� ��ȯ
    void SetGameMode(bool isSinglePlayer)
    {
        // �̱� ���� 1, ��Ƽ ���� 2 ����
        PlayerPrefs.SetInt("GameMode", isSinglePlayer ? 1 : 2);
        PlayerPrefs.Save();

        // GameScene���� �� ��ȯ
        SceneManager.LoadScene("SelectScene");
    }
}
