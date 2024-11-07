using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.UI; // UI ���� ���ӽ����̽�s
using UnityEngine.SceneManagement; // �� ��ȯ�� ���� ���ӽ����̽�

public class GameManager : MonoBehaviour
{

    public Button completeGameButton; // ���� �Ϸ� ��ư

    void Start()
    {
        // PlayerPrefs�� ����� ī�޶� �� ����� ������ �α׷� ���
        PrintAllPlayerPrefs();
        // ���� �Ϸ� ��ư Ŭ�� �̺�Ʈ ����
        completeGameButton.onClick.AddListener(CompleteGame);
    }



    // PlayerPrefs�� ����� ��� ������ ����ϴ� �Լ� (�ߺ� ����)
    void PrintAllPlayerPrefs()
    {
        // ����� GameMode �ҷ�����
        if (PlayerPrefs.HasKey("GameMode"))
        {
            int gameMode = PlayerPrefs.GetInt("GameMode");

            if (gameMode == 1)
            {
                Debug.Log("�̱��÷��� ���� ����");
            }
            else if (gameMode == 2)
            {
                Debug.Log("��Ƽ�÷��� ���� ����");
            }
            else
            {
                Debug.Log("�߸��� gameMode�Դϴ�.");
            }
        }
        else
        {
            Debug.Log("GameMode�� ã�� �� �����ϴ�.");
        }


        // ���õ� ī�޶� ���� ���
        if (PlayerPrefs.HasKey("SelectedCameraIndex"))
        {
                int selectedCameraIndex = PlayerPrefs.GetInt("SelectedCameraIndex");
                Debug.Log("SelectedCameraIndex: " + selectedCameraIndex);
        }
        else
        {
            Debug.Log("ī�޶� ������ ã�� �� �����ϴ�.");
        }

        // ���õ� ����� ���� ���
        if (PlayerPrefs.HasKey("SelectedMusicIndex"))
        {
            int selectedThumbnailIndex = PlayerPrefs.GetInt("SelectedMusicIndex");
            Debug.Log("SelectedMusicIndex: " + selectedThumbnailIndex);
        }
        else
        {
            Debug.Log("���� ������ ã�� �� �����ϴ�.");
        }

        Debug.Log("PlayerPrefs ����� ��� ������ ����߽��ϴ�.");
    }  
    // ���� �Ϸ� �� ResultScene���� �̵��ϴ� �Լ�
    public void CompleteGame()
    {
        // ResultScene���� ��ȯ
        SceneManager.LoadScene("ResultScene");
    }
}