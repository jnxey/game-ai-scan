import cv2
import numpy as np
import zxing
import tempfile
import os
import time
# python -m venv venv
# source venv/bin/activate

def locate_barcode_contour(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    gradX = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=-1)
    gradY = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=-1)
    gradient = cv2.subtract(gradX, gradY)
    gradient = cv2.convertScaleAbs(gradient)

    blur = cv2.blur(gradient, (9, 9))
    _, thresh = cv2.threshold(blur, 225, 255, cv2.THRESH_BINARY)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (21, 7))
    closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    closed = cv2.erode(closed, None, iterations=2)
    closed = cv2.dilate(closed, None, iterations=2)

    cnts, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return None

    return max(cnts, key=cv2.contourArea)


def order_points(pts):
    rect = np.zeros((4, 2), dtype="float32")

    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]   # 左上
    rect[2] = pts[np.argmax(s)]   # 右下

    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]  # 右上
    rect[3] = pts[np.argmax(diff)]  # 左下

    return rect


def warp_barcode(img, contour):
    rect = cv2.minAreaRect(contour)
    box = cv2.boxPoints(rect)
    box = np.array(box, dtype="float32")

    box = order_points(box)

    (tl, tr, br, bl) = box
    widthA = np.linalg.norm(br - bl)
    widthB = np.linalg.norm(tr - tl)
    maxWidth = int(max(widthA, widthB))

    heightA = np.linalg.norm(tr - br)
    heightB = np.linalg.norm(tl - bl)
    maxHeight = int(max(heightA, heightB))

    dst = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]], dtype="float32")

    M = cv2.getPerspectiveTransform(box, dst)
    warped = cv2.warpPerspective(img, M, (maxWidth, maxHeight))

    # 保证横向（宽 > 高）
    if warped.shape[0] > warped.shape[1]:
        warped = cv2.rotate(warped, cv2.ROTATE_90_CLOCKWISE)

    return warped


# -----------------------------
# 主流程
# -----------------------------
t1 = time.time()
img = cv2.imread("barcode2.png")

cnt = locate_barcode_contour(img)
if cnt is None:
    print("未找到条形码轮廓")
    exit()

warped = warp_barcode(img, cnt)

t2 = time.time()

# 把图像临时写成文件供 ZXing 处理
with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
    path = tmp.name
    cv2.imwrite(path, warped)

reader = zxing.BarCodeReader()
result = reader.decode(path)

t3 = time.time()

print("定位耗时:", t2 - t1)
print("解析耗时:", t3 - t2)

# 打印识别结果
if result and result.parsed:
    print("格式:", result.format)
    print("内容:", result.parsed)
else:
    print("未识别到条码")

# 清理临时文件
try:
    os.remove(path)
except:
    pass
