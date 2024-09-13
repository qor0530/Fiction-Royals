using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.SceneManagement; // 씬 전환을 위한 네임스페이스
using UnityEngine.UI; // UI 버튼을 사용하기 위한 네임스페이스
using System.Collections.Generic; // List 사용을 위해

public class SceneChange : MonoBehaviour
{
    // Button과 SceneName을 저장할 클래스를 만듦
    [System.Serializable]
    public class ButtonScenePair
    {
        public Button button; // UI 버튼
        public string sceneName; // 전환할 씬 이름
    }

    // 리스트로 여러 개의 버튼-씬 연결을 저장
    public List<ButtonScenePair> buttonScenePairs;

    void Start()
    {
        // 각 버튼에 대한 클릭 이벤트를 설정
        foreach (var pair in buttonScenePairs)
        {
            pair.button.onClick.AddListener(() => ChangeScene(pair.sceneName));
        }
    }

    // 씬을 전환하는 메서드
    void ChangeScene(string sceneName)
    {
        SceneManager.LoadScene(sceneName);
    }
}
