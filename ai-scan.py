from fastapi import FastAPI, File, UploadFile
from PIL import Image
import uvicorn
from ultralytics import YOLO
import time
import io
from card_matcher import format_poker_detections, format_majiang_detections
import requests
import asyncio
import httpx

PORT = 9981

app = FastAPI()

pokerModel = YOLO('poker-best8m.pt')

majiangModel = YOLO('majiang-best8m.pt')

@app.get("/check")
def check():
    return {"status": "success"}


@app.post("/poker-scan")
async def poker_scan(file: UploadFile = File(...)):
    try:
        # 读取图片字节
        img_bytes = await file.read()
        img = Image.open(io.BytesIO(img_bytes)).convert("RGB")  # 转为 RGB
        t1 = time.time()
        # YOLO 可以直接传入 PIL Image 或 numpy array
        results = pokerModel.predict(source=img, data='data.yaml', conf=0.7, device='cpu', save=False, show=False)  # 可调参数
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
        results = majiangModel.predict(source=img, data='data.yaml', conf=0.7, device='cpu', save=False, show=False)  # 可调参数
        print("YOLO耗时:", time.time() - t1)
        # 解析结果
        detections = format_majiang_detections(results)
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
