from fastapi import FastAPI, File, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from PIL import Image
import uvicorn
from ultralytics import YOLO
import time
import io
from card_matcher import format_poker_detections, format_majiang_detections, format_chip_detections
from chip_matcher import process_image, recognize_chip, ensure_cv2_image
import requests
import asyncio
import httpx
import torch

PORT = 9981

app = FastAPI()

# 配置 CORS
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"],
                   allow_headers=["*"], )

pokerModel = YOLO('poker-best8m.pt')

majiangModel = YOLO('majiang-best8m.pt')

chipModel = YOLO('chips-best8m.pt')

# 热身推理一次，丢一张小图
pokerModel("prerun.png", imgsz=416)
majiangModel("prerun.png", imgsz=416)
chipModel("prerun.png", imgsz=416)


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
        results = pokerModel.predict(source=img, data='data.yaml', conf=0.7, device=0, save=False, show=False)  # 可调参数
        print("YOLO耗时:", time.time() - t1)
        # 解析结果
        detections = format_poker_detections(results)
        return {"code": 1, "data": detections, "msg": "ok"}
    except Exception as e:
        return {"code": 0, "msg": "推理异常"}


@app.post("/majiang-scan")
async def poker_scan(file: UploadFile = File(...)):
    try:
        # 读取图片字节
        img_bytes = await file.read()
        img = Image.open(io.BytesIO(img_bytes)).convert("RGB")  # 转为 RGB
        t1 = time.time()
        # YOLO 可以直接传入 PIL Image 或 numpy array
        results = majiangModel.predict(source=img, data='data.yaml', conf=0.7, device=0, save=False, show=False)  # 可调参数
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
        results = chipModel.predict(source=img, data='data.yaml', conf=0.7, device=0, save=False, show=False)  # 可调参数
        t2 = time.time()
        print("YOLO耗时:", t2 - t1)
        # 解析结果
        detections = format_chip_detections(results)
        if scan_text == 'yes':
            for det in detections:
                chip_img = process_image(ensure_cv2_image(img), det['bbox'])
                view = recognize_chip(chip_img)
                det['view'] = view
        t3 = time.time()
        print("OCR耗时:", t3 - t2)
        return {"code": 1, "data": detections, "msg": "ok"}

    except Exception as e:
        print(e)
        return {"code": 0, "msg": "推理异常"}


if __name__ == "__main__":
    uvicorn.run(
        "ai-scan:app",  # 等价于命令行 main:app
        host="0.0.0.0",
        port=PORT,
        workers=1,  # Windows 必须 = 1
        reload=False,  # 一定要关
        log_level="info"
    )
