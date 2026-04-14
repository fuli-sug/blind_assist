import flet as ft
import cv2
import numpy as np
import onnxruntime as ort
import threading
import time

# ================== 加载 ONNX 模型 ==================
# 确保 model.onnx 文件在项目根目录
sess = ort.InferenceSession("model.onnx")

# 获取输入输出信息（可选）
input_name = sess.get_inputs()[0].name
output_name = sess.get_outputs()[0].name

class_names = ["100yuan", "10yuan", "1yuan", "20yuan", "50yuan", "5yuan"]

def preprocess_image(frame):
    """预处理图像：resize + 归一化 + 转换维度 (NHWC -> NCHW)"""
    img = cv2.resize(frame, (224, 224))
    img = img.astype(np.float32) / 255.0
    # 添加 batch 维度并转换通道顺序 (H, W, C) -> (1, C, H, W)
    img = np.expand_dims(img, axis=0).transpose(0, 3, 1, 2)
    return img

def predict(frame):
    """推理"""
    input_data = preprocess_image(frame)
    outputs = sess.run([output_name], {input_name: input_data})
    pred_idx = np.argmax(outputs[0][0])
    return class_names[pred_idx]

# ================== UI界面 ==================
def main(page: ft.Page):
    page.title = "助盲识别系统"
    page.theme_mode = ft.ThemeMode.DARK
    page.padding = 20

    title = ft.Text("人民币识别系统", size=28, weight=ft.FontWeight.BOLD)
    result_text = ft.Text("等待识别...", size=32, weight=ft.FontWeight.BOLD, color=ft.Colors.GREEN)
    status_text = ft.Text("状态: 未开始", size=14, color=ft.Colors.GREY)

    page.add(
        ft.Column([
            title,
            ft.Divider(),
            result_text,
            ft.Container(height=20),
            status_text,
            ft.ElevatedButton("开始识别", on_click=lambda e: start_recognition(page, result_text, status_text))
        ], alignment=ft.MainAxisAlignment.CENTER)
    )

def start_recognition(page: ft.Page, result_text, status_text):
    status_text.value = "状态: 识别中..."
    page.update()
    threading.Thread(target=recognition_loop, args=(page, result_text, status_text), daemon=True).start()

def recognition_loop(page: ft.Page, result_text, status_text):
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        status_text.value = "状态: 摄像头打开失败"
        page.update()
        return

    last_spoken = None
    stable_count = 0
    last_detected = None

    while True:
        ret, frame = cap.read()
        if not ret:
            continue

        # 推理
        detected = predict(frame)

        # 稳定判断
        if detected == last_detected:
            stable_count += 1
        else:
            stable_count = 1
            last_detected = detected

        if stable_count >= 3:
            show_text = detected.replace("yuan", "元")
            result_text.value = f"识别到{show_text}"
            page.update()

            # 语音播报
            if last_spoken != detected:
                speak(show_text)
                last_spoken = detected
        else:
            # 可选：显示正在稳定中
            result_text.value = "未稳定..."
            page.update()

        time.sleep(0.1)

    cap.release()

def speak(text):
    """语音播报（Termux 或 Android）"""
    try:
        import subprocess
        subprocess.run(['termux-tts-speak', text])
    except:
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

if __name__ == "__main__":
    ft.app(target=main)
