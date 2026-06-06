from datetime import date, datetime

from fastapi import FastAPI, File, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from PIL import Image
import uvicorn
from ultralytics import YOLO
import time
import io
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse
from custom_cipher import decode
from card_matcher import format_poker_detections, format_majiang_detections, format_chip_detections
from chip_matcher import process_image, recognize_chip
from chip_ocr_text import detect_and_crop
from chip_ocr_easy import easyocr_digits_only, preprocess_for_ocr, process_chip_image, dummy_prev
import requests
import asyncio
import httpx
import torch
import cv2
import os

# 固定线程，稳定性能
torch.set_num_threads(2)
torch.set_num_interop_threads(1)
cv2.setNumThreads(2)
cv2.ocl.setUseOpenCL(False)

DEVICE = 0 if torch.cuda.is_available() else "cpu"
print(f"推理设备: {DEVICE}")

PORT = 9981

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CERT_DIR = os.path.join(BASE_DIR, "certs")
SSL_CERTFILE = os.environ.get("HTTPS_CERT", os.path.join(CERT_DIR, "localhost.pem"))
SSL_KEYFILE = os.environ.get("HTTPS_KEY", os.path.join(CERT_DIR, "localhost-key.pem"))

CHECK_CHAR_HEADER = "x-check-char"
CHECK_CHAR_KEY = os.environ.get("CHECK_CHAR_KEY", "gv")
MAX_DATE_DIFF_DAYS = 2
AUTH_SKIP_PATHS = {"/check", "/demo"}


def _verify_check_char(header_value: str | None) -> bool:
    if not header_value or not header_value.strip():
        return False
    try:
        plain = decode(header_value.strip(), CHECK_CHAR_KEY)
        client_date = datetime.strptime(plain.strip(), "%Y%m%d").date()
    except (ValueError, Exception):
        return False
    return abs((date.today() - client_date).days) <= MAX_DATE_DIFF_DAYS


class CheckCharMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        if request.method == "OPTIONS":
            return await call_next(request)
        if request.method == "GET" and request.url.path in AUTH_SKIP_PATHS:
            return await call_next(request)
        if not _verify_check_char(request.headers.get(CHECK_CHAR_HEADER)):
            return JSONResponse(status_code=401, content={"code": 0, "msg": "鉴权失败"})
        return await call_next(request)


app = FastAPI()

# CheckChar 先注册，CORS 后注册，保证 401 响应仍带 CORS 头
app.add_middleware(CheckCharMiddleware)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"],
                   allow_headers=["*"], )

pokerModel = YOLO('poker-best8m.pt')

majiangModel = YOLO('majiang-best8m.pt')

chipModel = YOLO('chips-best8m.pt')

# 热身推理一次，丢一张小图
pokerModel("prerun.png", imgsz=416, device=DEVICE)
majiangModel("prerun.png", imgsz=416, device=DEVICE)
chipModel("prerun.png", imgsz=416, device=DEVICE)

@app.get("/check")
def check():
    return {"status": "success"}


@app.get("/demo", response_class=HTMLResponse)
async def read_root():
    with open("templates/index.html", "r", encoding="utf-8") as f:
        html_content = f.read()
    return HTMLResponse(content=html_content)


@app.post("/poker-scan")
async def poker_scan(file: UploadFile = File(...)):
    try:
        # 读取图片字节
        img_bytes = await file.read()
        img = Image.open(io.BytesIO(img_bytes)).convert("RGB")  # 转为 RGB
        t1 = time.time()
        # YOLO 可以直接传入 PIL Image 或 numpy array
        results = pokerModel.predict(source=img, data='data.yaml', conf=0.7, device=DEVICE, save=False, show=False)  # 可调参数
        print("YOLO耗时:", time.time() - t1)
        # 解析结果
        detections = format_poker_detections(results)
        return {"code": 1, "data": detections, "msg": "ok"}
    except Exception as e:
        print("-----------------error-----------------------------")
        print(e)
        print("-----------------error-----------------------------")
        return {"code": 0, "msg": "推理异常"}


@app.post("/majiang-scan")
async def poker_scan(file: UploadFile = File(...)):
    try:
        # 读取图片字节
        img_bytes = await file.read()
        img = Image.open(io.BytesIO(img_bytes)).convert("RGB")  # 转为 RGB
        t1 = time.time()
        # YOLO 可以直接传入 PIL Image 或 numpy array
        results = majiangModel.predict(source=img, data='data.yaml', conf=0.7, device=DEVICE, save=False, show=False)  # 可调参数
        print("YOLO耗时:", time.time() - t1)
        # 解析结果
        detections = format_majiang_detections(results)
        return {"code": 1, "data": detections, "msg": "ok"}
    except Exception as e:
        print(e)
        return {"code": 0, "msg": "推理异常"}


@app.post("/chip-scan")
async def chip_scan(file: UploadFile = File(...), scan_text: str = Form(...), ):
    try:
        # 读取图片字节
        img_bytes = await file.read()
        img = Image.open(io.BytesIO(img_bytes)).convert("RGB")  # 转为 RGB

        t1 = time.time()
        # YOLO 可以直接传入 PIL Image 或 numpy array
        results = chipModel.predict(source=img, data='data.yaml', conf=0.7, device=DEVICE, save=False, show=False)  # 可调参数
        t2 = time.time()
        if scan_text == 'yes':
            print("YOLO耗时:", t2 - t1)
        # 解析结果
        detections = format_chip_detections(results)
        try:
            if scan_text == 'yes':
                for det in detections:
                    time.sleep(0.002) # 休息2ms，给CPU，GPU点时间缓缓
                    chip_img = process_chip_image(img, det['bbox'], pad_ratio_w=0, pad_ratio_h=0, pad_bottom=False)
                    # cv2.imshow('chip_img', chip_img)
                    # cv2.waitKey(0)
                    roi, angle = detect_and_crop(chip_img)
                    if roi is None or roi.size == 0:
                        print("⚠️ ROI 为空，无法进行 OCR")
                    else:
                        code = easyocr_digits_only(preprocess_for_ocr(roi))
                        if code is not None:
                            det['view'] = {"code": code, "angle": angle}
            t3 = time.time()
            if scan_text == 'yes':
                print("OCR耗时:", t3 - t2)
        except Exception as e:
            print(e)
        return {"code": 1, "data": detections, "msg": "ok"}
    except Exception as e:
        print(e)
        return {"code": 0, "msg": "推理异常"}


if __name__ == "__main__":
    if not os.path.exists(SSL_CERTFILE) or not os.path.exists(SSL_KEYFILE):
        raise FileNotFoundError(
            f"TLS 证书不存在，请将证书放入 ./certs 或通过 HTTPS_CERT / HTTPS_KEY 指定路径:\n"
            f"  {SSL_CERTFILE}\n"
            f"  {SSL_KEYFILE}"
        )

    print(f"服务启动: https://127.0.0.1:{PORT}")
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=PORT,
        ssl_certfile=SSL_CERTFILE,
        ssl_keyfile=SSL_KEYFILE,
        workers=1,  # Windows 必须 = 1
        reload=False,  # 一定要关
        log_level="info"
    )
