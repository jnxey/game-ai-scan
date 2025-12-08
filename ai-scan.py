# app_flask.py
from flask import Flask, request, redirect, url_for, render_template, flash, jsonify
import socket
import sys
import tkinter.messagebox as mb
import os
import socket
import shutil
from PIL import Image
import io
from ultralytics import YOLO
import time

PORT = 9981

APP_NAME = "ai-scan.exe"

yoloModel = YOLO('poker-best8m.pt')

# =========================
# 1. 检测是否端口被占用
# =========================
def prevent_multi_instance(port):
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)

    try:
        sock.bind(("127.0.0.1", port))
        return sock
    except OSError:
        # 端口被占用，查询 PID
        for conn in psutil.net_connections(kind='inet'):
            if conn.laddr.port == port:
                pid = conn.pid
                if pid == os.getpid():
                    # 占用的是自己进程，允许继续
                    return sock
                else:
                    print(f"端口 {port} 被 PID {pid} 占用，程序已启动！")
                    input("按回车退出...")
                    sys.exit(1)
        # 没查到 PID，也直接退出
        print(f"端口 {port} 被占用，程序已启动！")
        input("按回车退出...")
        sys.exit(1)

if __name__ == "__main__":
    lock = prevent_multi_instance(PORT)
    print(f"程序启动成功，占用端口 {PORT}")

# =========================
# 2. 设置开机启动
# =========================
def set_autostart():
    if not getattr(sys, 'frozen', False):
        print("🧪 当前为 python 运行模式，不设置开机启动")
        return
    exe = sys.executable
    startup = os.path.join(
        os.environ["APPDATA"],
        r"Microsoft\Windows\Start Menu\Programs\Startup"
    )
    target = os.path.join(startup, APP_NAME)
    if not os.path.exists(target):
        shutil.copyfile(exe, target)
        print("✅ 已自动加入开机启动")
    else:
        print("✅ 开机启动已存在")

set_autostart()

# =========================
# 3. 主业务
# =========================

app = Flask(__name__)

app.secret_key = 'supersecretkey'  # 用于 flash 消息

# 设置上传文件夹
UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# 允许的文件扩展名
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}

# 检查文件是否允许上传
def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route("/")
def index():
    return "Hello, HTTP Server!"

@app.route("/check", methods=['GET'])
def check():
    return "success"

@app.route("/demo", methods=['GET'])
def demo():
    return render_template('index.html')

@app.route('/poker-scan', methods=['POST'])
def poker_scan():
    if 'file' not in request.files:
        return jsonify({"error": "没有上传文件"}), 400

    file = request.files['file']

    if file.filename == '':
        return jsonify({"error": "未选择文件"}), 400

    # 将上传的文件读取为 PIL Image
    img_bytes = file.read()
    img = Image.open(io.BytesIO(img_bytes)).convert("RGB")  # 转为 RGB
    # img = cv2.imdecode(np.frombuffer(img_bytes, np.uint8), cv2.IMREAD_COLOR)

    t1 = time.time()
    # YOLO 可以直接传入 PIL Image 或 numpy array
    results = yoloModel.predict(source=img,data='data.yaml',conf=0.7,device='cpu',save=False,show=False)  # 可调参数
    print("YOLO耗时:", time.time()-t1)

    # 解析结果
    detections = []
    for r in results:
        boxes = r.boxes.xyxy.tolist()  # [[x1, y1, x2, y2], ...]
        scores = r.boxes.conf.tolist()  # 置信度
        classes = r.boxes.cls.tolist()  # 类别索引
        for b, s, c in zip(boxes, scores, classes):
            detections.append({
                "box": b,
                "score": s,
                "class_id": int(c)
            })
    return jsonify({"detections": detections})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=PORT, debug=False)
