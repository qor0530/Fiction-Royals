using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.SceneManagement; // �� ��ȯ�� ���� ���ӽ����̽�
using UnityEngine.UI; // UI ��ư�� ����ϱ� ���� ���ӽ����̽�
using System.Collections.Generic; // List ����� ����

public class SceneChange : MonoBehaviour
{
    // Button�� SceneName�� ������ Ŭ������ ����
    [System.Serializable]
    public class ButtonScenePair
    {
        public Button button; // UI ��ư
        public string sceneName; // ��ȯ�� �� �̸�
    }

    // ����Ʈ�� ���� ���� ��ư-�� ������ ����
    public List<ButtonScenePair> buttonScenePairs;

    void Start()
    {
        // �� ��ư�� ���� Ŭ�� �̺�Ʈ�� ����
        foreach (var pair in buttonScenePairs)
        {
            pair.button.onClick.AddListener(() => ChangeScene(pair.sceneName));
        }
    }

    // ���� ��ȯ�ϴ� �޼���
    void ChangeScene(string sceneName)
    {
        SceneManager.LoadScene(sceneName);
    }
}
