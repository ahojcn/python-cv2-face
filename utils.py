import sqlite3
import cv2
import base64
import numpy as np
import pygame
import os
from PIL import Image, ImageDraw, ImageFont
from aip import AipFace, AipSpeech

BAIDU_AI_APP_ID = "18145377"
BAIDU_AI_API_KEY = "QdG40KGS1lfADSR9yYqVcNVG"
BAIDU_AI_SECRET_KEY = "ATDcjp1Ex62LnclBY7tl04uDY14Nyt1u"


class DBOperator(object):
    def __init__(self, db_name="face.db"):
        self._conn = sqlite3.connect(db_name)
        self._c = self._conn.cursor()
        print(f"[INFO] 初始化数据库{db_name}成功")

    def user_add(self, name):
        """
        添加一个用户
        """
        try:
            self._c.execute(f"insert into user (name ) values ({name})")
        except Exception as e:
            print(f"[ERROR] {e}")

    def user_count_add(self, name):
        """
        给name用户的count+1
        """
        try:
            cursor = self._c.execute(f"select id, name, count from user where name='{name}'")
            index, name, count = None, None, None
            for row in cursor:
                index = row[0]
                name = row[1]
                count = int(row[2])
            print(f"[INFO] 根据名称检索出用户 {index},{name},{count}")

            self._c.execute(f"update user set count = {count + 1} where id = {index} and name = {name}")
            self._conn.commit()
            print(f"[INFO] 更新用户count成功，影响{self._conn.total_changes}行")
        except Exception as e:
            print(f"[ERROR] {e}")


class BaiduAPI(object):
    def __init__(self):
        self._face_cli = AipFace(BAIDU_AI_APP_ID, BAIDU_AI_API_KEY, BAIDU_AI_SECRET_KEY)
        self._speech_cli = AipSpeech(BAIDU_AI_APP_ID, BAIDU_AI_API_KEY, BAIDU_AI_SECRET_KEY)

    def get_face_cli(self):
        return self._face_cli

    def get_speech_cli(self):
        return self._speech_cli

    def face_search(self, frame, group_id_list="ahojcn"):
        image_base64 = image_to_base64(frame)
        resp = self._face_cli.search(image_base64, "BASE64", group_id_list)
        result = None
        try:
            if resp["error_msg"] == "SUCCESS":
                result = resp["result"]["user_list"][0]
                print(result)
        except Exception as e:
            print(f"[ERROR] {e}")
            raise e
        return result

    def audio_speech(self, text):
        result = self._speech_cli.synthesis(text, options={'vol': 15})
        if not isinstance(result, dict):
            with open("audio.mp3", "wb") as f:
                f.write(result)

        # 此处可以使用阻塞队列来进行管理播放队列！
        if not isinstance(result, dict):
            pygame.mixer.init()
            pygame.mixer.music.load("audio.mp3")
            pygame.mixer.music.play()
            pygame.mixer.music.stop()


def image_to_base64(image_np):
    image = cv2.imencode('.jpg', image_np)[1]
    image_code = str(base64.b64encode(image))[2:-1]

    return image_code


def image_draw_text(img, text, left, top, textColor=(0, 255, 0), textSize=20):
    if isinstance(img, np.ndarray):  # 判断是否OpenCV图片类型
        img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    # 创建一个可以在给定图像上绘图的对象
    draw = ImageDraw.Draw(img)
    # 字体的格式
    fontStyle = ImageFont.truetype(
        "simsun.ttc", textSize, encoding="utf-8")
    # 绘制文本
    draw.text((left, top), text, textColor, font=fontStyle)
    # 转换回OpenCV格式
    return cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)


def face_search():
    classifier = cv2.CascadeClassifier("./haarcascade_frontalface_default.xml")
    color = (255, 0, 0)
    cv2.namedWindow("Live", cv2.WINDOW_NORMAL)
    baidu = BaiduAPI()

    cap = cv2.VideoCapture(0)
    while True:
        ret, frame = cap.read()
        # cv2.imwrite("1.png", frame)
        face_rects = classifier.detectMultiScale(frame, scaleFactor=1.2, minNeighbors=3, minSize=(32, 32))
        print(len(face_rects))
        for f in face_rects:
            x, y, w, h = f
            cv2.rectangle(frame, (x, y), (x + h, y + w), color, 2)
            cv2.circle(frame, (x + w // 4, y + h // 4 + 30), min(w // 8, h // 8), color)
            cv2.circle(frame, (x + 3 * w // 4, y + h // 4 + 30), min(w // 8, h // 8), color)
            cv2.rectangle(frame, (x + 3 * w // 8, y + 3 * h // 4), (x + 5 * w // 8, y + 7 * h // 8), color)
        try:
            result = baidu.face_search(frame)
            frame = image_draw_text(frame, result["user_info"], x + w / 2, y, (255, 0, 0), int(h / 10))
        except Exception as e:
            print(f"[ERROR] {e}")
        # cv2.imwrite("image.png", frame)
        cv2.imshow("Live", frame)
        # break
        if cv2.waitKey(100) & 0xff == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()


def photo_face_search():
    classifier = cv2.CascadeClassifier("./haarcascade_frontalface_default.xml")
    color = (255, 0, 0)
    baidu = BaiduAPI()
    for image_name in os.listdir("./img"):
        frame = cv2.imread(f"./img/{image_name}")
        cv2.imshow(f"{image_name}", frame)
        face_rects = classifier.detectMultiScale(frame, scaleFactor=1.2, minNeighbors=3, minSize=(32, 32))
        for f in face_rects:
            x, y, w, h = f
            cv2.rectangle(frame, (x, y), (x + h, y + w), color, 2)
            cv2.circle(frame, (x + w // 4, y + h // 4 + 30), min(w // 8, h // 8), color)
            cv2.circle(frame, (x + 3 * w // 4, y + h // 4 + 30), min(w // 8, h // 8), color)
            cv2.rectangle(frame, (x + 3 * w // 8, y + 3 * h // 4), (x + 5 * w // 8, y + 7 * h // 8), color)
        result = baidu.face_search(frame)
        frame_result = image_draw_text(frame, result["user_info"], x + w / 2, y, (255, 0, 0), int(h / 10))
        cv2.imshow(f"{image_name}_result", frame_result)

    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == '__main__':
    face_search()
    # photo_face_search()
