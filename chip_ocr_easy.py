import cv2
import numpy as np
from PIL import Image
import time

from ocr_reader import get_reader

reader = get_reader()


# 预热图片
dummy_img = np.zeros((32, 128), dtype=np.uint8)

# 预热操作
def dummy_prev():
    return reader.readtext(dummy_img, detail=0)

# =========================
# 确保输入为 OpenCV ndarray
# =========================
def ensure_cv2_image(img):
    if isinstance(img, np.ndarray):
        return img
    if isinstance(img, Image.Image):
        img = np.array(img)
        # RGB -> BGR
        if img.ndim == 3 and img.shape[2] == 3:
            img = img[:, :, ::-1]
        return img
    raise ValueError("输入必须是 ndarray 或 PIL Image")

# =========================
# 裁剪图片（左/上可加 padding，底部保持 bbox 原样）
# =========================
def crop_chip_roi(img, x1, y1, x2, y2, pad_ratio_w=0.15, pad_ratio_h=0.15, pad_bottom=False):
    """
    img: ndarray (BGR)
    x1,y1,x2,y2: YOLO bbox 坐标
    pad_ratio_w: 左右扩展比例
    pad_ratio_h: 上方扩展比例
    pad_bottom: True 则底部也加 padding，否则保持原 bbox
    """
    h_img, w_img = img.shape[:2]

    bw = x2 - x1
    bh = y2 - y1

    pad_w = int(bw * pad_ratio_w)
    pad_h = int(bh * pad_ratio_h)

    nx1 = max(0, int(x1) - pad_w)
    nx2 = min(w_img, int(x2) + pad_w)

    ny1 = max(0, int(y1) - pad_h)
    ny2 = min(h_img, int(y2) + pad_h) if pad_bottom else int(y2)

    # 防止裁剪越界
    if nx2 <= nx1 or ny2 <= ny1:
        return None

    roi = img[ny1:ny2, nx1:nx2]
    roi = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
    return roi

# =========================
# 预处理：灰度 + 二值
# =========================
def preprocess(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    bin_img = cv2.adaptiveThreshold(
        gray, 255,
        cv2.ADAPTIVE_THRESH_MEAN_C,
        cv2.THRESH_BINARY_INV,
        31, 5
    )
    cv2.imshow("TTT", bin_img)
    cv2.waitKey(0)
    return bin_img

# =========================
# 主处理函数
# =========================
def process_chip_image(img, bbox, pad_ratio_w, pad_ratio_h, pad_bottom=False):
    """
    img: PIL.Image 或 ndarray
    bbox: dict {"x1":.., "y1":.., "x2":.., "y2":..}
    """
    img_cv = ensure_cv2_image(img)
    roi = crop_chip_roi(
        img_cv,
        bbox["x1"], bbox["y1"], bbox["x2"], bbox["y2"],
        pad_ratio_w=pad_ratio_w,
        pad_ratio_h=pad_ratio_h,
        pad_bottom=pad_bottom
    )
    if roi is None or roi.size == 0:
        raise ValueError("裁剪失败，ROI为空")
    return roi

def preprocess_for_ocr(roi):
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    _, th = cv2.threshold(gray, 0, 255,
                          cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    resized = normalize_roi(th)
    return resized

# =========================
# 统一宽度
# =========================
def normalize_roi(gray, target_h=48, max_w=150):
    h, w = gray.shape
    # 等比例缩放到目标高度
    scale = target_h / h
    new_w = int(w * scale)
    # 限制最大宽度（关键）
    if new_w > max_w:
        new_w = max_w
    resized = cv2.resize(gray, (new_w, target_h))
    # cv2.imshow("TTT", resized)
    # cv2.waitKey(0)
    return resized

def easyocr_digits_only(img):
    t1 = time.time()
    results = reader.readtext(img, detail=1, paragraph=False)
    print("Reader耗时:", time.time() - t1)
    digits = []
    for _, text, conf in results:
        # print(text,'------------------text')
        # print(conf,'------------------conf')
        if conf > 0.82 and len(text) == 6:
            digits.append(text)
    if not digits:
        print("⚠️ EasyOCR 未识别到数字")
        return None
    return digits[0]

# img = cv2.imread("easy_ocr4.png")
# t1 = time.time()
# result = easyocr_digits_only(preprocess(img))
# print("OCR耗时:", time.time() - t1)
# print("最终结果：", result)
