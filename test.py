import cv2

import utils


def test_baidu_face_api():
    img = cv2.imread("img/leijinpeng.jpg")
    baidu = utils.BaiduAPI()
    ret = baidu.get_face_cli().face_search(utils.image_to_base64(img), "BASE64", "ahojcn")
    print(ret)


def test_baidu_speech_api():
    cli = utils.BaiduAPI().get_speech_cli()
    result = cli.synthesis("你好百度", 'zh', 1, {'vol': 15})
    if not isinstance(result, dict):
        with open('audio.mp3', 'wb') as f:
            f.write(result)


if __name__ == '__main__':
    # test_baidu_face_api()
    # test_baidu_speech_api()
    pass
    # import pygame
    #
    # pygame.mixer.init()  # 初始化
    # pygame.mixer.music.load("audio.mp3")
    # # pygame.mixer.music.play()
    # import multiprocessing
    #
    # multiprocessing.Process(target=pygame.mixer.music.play())
    #
    # while True:
    #     print(1)
