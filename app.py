import flet as ft
import cv2
import torch
import torch.nn as nn
import torchvision.models as models
from torchvision import transforms
from PIL import Image
from ultralytics import YOLO
import threading
import time


# ================== 语音播报 ==================
def speak(text):
    """Android TTS 语音播报"""
    try:
        from jnius import autoclass
        PythonActivity = autoclass('org.kivy.android.PythonActivity')
        TextToSpeech = autoclass('android.speech.tts.TextToSpeech')
        tts = TextToSpeech(PythonActivity.mActivity, None)
        Locale = autoclass('java.util.Locale')
        tts.setLanguage(Locale.CHINESE)
        tts.speak(text, TextToSpeech.QUEUE_FLUSH, None)
    except:
        print(f"[语音] {text}")


# ================== 加载模型 ==================
# 注意：APK打包后，模型文件需要放在正确位置
model = YOLO("best.pt")  # 模型文件要放到项目目录

classifier = models.mobilenet_v2(weights=None)
classifier.classifier[1] = nn.Linear(classifier.last_channel, 6)
classifier.load_state_dict(torch.load("classifier.pth", map_location="cpu"))
classifier.eval()

class_names = ["100yuan", "10yuan", "1yuan", "20yuan", "50yuan", "5yuan"]

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])


# ================== UI界面 ==================
def main(page: ft.Page):
    page.title = "助盲识别系统"
    page.theme_mode = ft.ThemeMode.DARK
    page.padding = 20
    page.window_width = 400
    page.window_height = 700

    # 标题
    title = ft.Text("人民币识别系统", size=28, weight=ft.FontWeight.BOLD)

    # 识别结果显示
    result_text = ft.Text("等待识别...", size=32, weight=ft.FontWeight.BOLD, color=ft.Colors.GREEN)

    # 状态显示
    status_text = ft.Text("状态: 未开始", size=14, color=ft.Colors.GREY)

    # 语音开关
    voice_switch = ft.Switch(label="语音播报", value=True)

    # 识别按钮
    start_btn = ft.ElevatedButton("开始识别", icon=ft.icons.CAMERA,
                                  on_click=lambda e: start_recognition(page, result_text, status_text, voice_switch))

    page.add(
        ft.Column([
            title,
            ft.Divider(),
            result_text,
            ft.Container(height=20),
            status_text,
            voice_switch,
            start_btn
        ], alignment=ft.MainAxisAlignment.CENTER, horizontal_alignment=ft.CrossAxisAlignment.CENTER)
    )


def start_recognition(page: ft.Page, result_text, status_text, voice_switch):
    """开始识别（在新线程中运行）"""
    status_text.value = "状态: 识别中..."
    page.update()
    threading.Thread(target=recognition_loop, args=(page, result_text, status_text, voice_switch), daemon=True).start()


def recognition_loop(page: ft.Page, result_text, status_text, voice_switch):
    """识别循环"""
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        status_text.value = "状态: 摄像头打开失败"
        page.update()
        return

    status_text.value = "状态: 识别中，请对准纸币..."
    page.update()

    last_spoken = None
    stable_count = 0
    last_detected = None

    while True:
        ret, frame = cap.read()
        if not ret:
            continue

        # YOLO检测
        results = model(frame, conf=0.6)

        detected = None

        for r in results:
            if r.boxes is None:
                continue

            for box in r.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])

                # ROI扩展
                pad = 10
                x1 = max(0, x1 - pad)
                y1 = max(0, y1 - pad)
                x2 = min(frame.shape[1], x2 + pad)
                y2 = min(frame.shape[0], y2 + pad)

                roi = frame[y1:y2, x1:x2]
                if roi.size == 0:
                    continue

                # 分类
                img = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
                img = Image.fromarray(img)
                img = transform(img).unsqueeze(0)

                with torch.no_grad():
                    output = classifier(img)
                    pred = torch.argmax(output, dim=1).item()

                final_label = class_names[pred]

                # 稳定判断
                if final_label == last_detected:
                    stable_count += 1
                else:
                    stable_count = 1
                    last_detected = final_label

                if stable_count >= 3:
                    detected = final_label

        if detected:
            show_text = detected.replace("yuan", "元")
            result_text.value = f"识别到{show_text}"
            page.update()

            # 语音播报
            if voice_switch.value:
                current_time = time.time()
                if last_spoken != detected:
                    speak(f"识别到{show_text}")
                    last_spoken = detected
        else:
            result_text.value = "未识别到纸币"
            page.update()

        time.sleep(0.1)

    cap.release()


# ================== 打包入口 ==================
if __name__ == "__main__":
    ft.app(target=main)