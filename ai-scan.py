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
from card_matcher import format_poker_detections, format_majiang_detections

PORT = 9981

APP_NAME = "ai-scan.exe"

pokerModel = YOLO('poker-best8m.pt')

majiangModel = YOLO('majiang-best8m.pt')

# =========================
# 1. 主业务
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


# 扑克牌扫描
@app.route('/poker-scan', methods=['POST'])
def poker_scan():
    try:
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
        results = pokerModel.predict(source=img, data='data.yaml', conf=0.7, device='cpu', save=False, show=False)  # 可调参数
        print("YOLO耗时:", time.time() - t1)

        # 解析结果
        detections = format_poker_detections(results)
        return jsonify({"code": 1, "data": detections, "msg": "ok"})
    except Exception as e:
        return jsonify({"code": 0, "msg": "推理异常"})


# 麻将扫描
@app.route('/majiang-scan', methods=['POST'])
def majiang_scan():
    try:
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
        results = majiangModel.predict(source=img, data='data.yaml', conf=0.5, device='cpu', save=False, show=False)  # 可调参数
        print("YOLO耗时:", time.time() - t1)

        # 解析结果
        detections = format_majiang_detections(results)
        return jsonify({"code": 1, "data": detections, "msg": "ok"})
    except Exception as e:
        return jsonify({"code": 0, "msg": "推理异常"})


@app.errorhandler(Exception)
def handle_exception(e):
    return f"服务器异常: {e}", 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=PORT, debug=False, use_reloader=False)
