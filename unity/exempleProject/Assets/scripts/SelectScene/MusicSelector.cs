using UnityEngine;
using UnityEngine.SceneManagement;
using UnityEngine.UI;
using System.Collections.Generic;

public class MusicSelector : MonoBehaviour
{
    // ����� �̹��� ����Ʈ (����Ƽ �����Ϳ��� �߰� ����)
    public List<Sprite> thumbnailList;
    
    // ������� ǥ���� UI Image
    public Image displayImage;
    
    // �¿� ȭ��ǥ ��ư
    public Button leftArrowButton;
    public Button rightArrowButton;

    // ���� ���õ� ����� �ε���
    private int currentIndex = 0;

    // ���õ� ������� Ŭ���ϸ� �� �̵��� �� �� �ֵ��� Button���� ����
    public Button thumbnailButton;

    void Start()
    {
        // �¿� ȭ��ǥ ��ư�� �̺�Ʈ ����
        leftArrowButton.onClick.AddListener(() => ChangeThumbnail(-1));
        rightArrowButton.onClick.AddListener(() => ChangeThumbnail(1));

        // ����� Ŭ�� �� ������ �����ϰ� �� ��ȯ
        thumbnailButton.onClick.AddListener(SelectMusic);

        // ù ��° ������� ������
        UpdateThumbnail();
    }

    // ������� �����ϴ� �Լ�
    void ChangeThumbnail(int direction)
    {
        // �ε��� ������Ʈ (����Ʈ�� ũ�⸦ ���� �ʵ��� ó��)
        currentIndex += direction;
        if (currentIndex < 0)
        {
            currentIndex = thumbnailList.Count - 1;
        }
        else if (currentIndex >= thumbnailList.Count)
        {
            currentIndex = 0;
        }

        // ������� ������Ʈ
        UpdateThumbnail();
    }

    // ���� �ε����� �´� ������� ȭ�鿡 ǥ��
    void UpdateThumbnail()
    {
        if (thumbnailList.Count > 0)
        {
            displayImage.sprite = thumbnailList[currentIndex];
        }
    }

    // ���õ� ������ �����ϰ� ������ ������ �� �ִ� �Լ�
    public void SelectMusic()
    {
        PlayerPrefs.SetInt("SelectedMusicIndex", currentIndex); // ���õ� ���� �ε����� ����
        PlayerPrefs.Save();

        SceneManager.LoadScene("FrameCheckScene");
    }
}
