using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class GameModeChecker : MonoBehaviour
{
    void Start()
    {
        // PlayerPrefs���� ���� ��� �� �ҷ����� (�⺻���� �̱��÷��� 1)
        int gameMode = PlayerPrefs.GetInt("GameMode", 1);

        if (gameMode == 1)
        {
            Debug.Log("�̱��÷��� ��� ����");
        }
        else
        {
            Debug.Log("��Ƽ�÷��� ��� ����");
        }
    }
}
