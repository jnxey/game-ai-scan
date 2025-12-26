import cv2
import numpy as np
import easyocr
import re
from PIL import Image

# 初始化 OCR（建议全局只初始化一次）
reader = easyocr.Reader(['en'], gpu=True, verbose=False)

# 合法面值（按你的筹码调整）
DENOMS = {'1', '5', '10', '25', '50', '100', '200', '500', '1000', '2000', '5000', '10000', '20000', '50000', '100000',
          '200000', '500000'}
# 码长度
CODE_LEN = 6


# 旋转图片
def rotate_image(image, angle):
    h, w = image.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    return cv2.warpAffine(
        image, M, (w, h),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_REPLICATE
    )


# 是否是code
def is_code(text):
    """
    判断是否像 code（固定长度）
    """
    text = text.replace(' ', '')
    return len(text) == CODE_LEN


# 找到面值和code
def find_value_and_code(image, conf_thresh=0.85):
    results = reader.readtext(image, detail=1)
    denom = None
    code = None
    for bbox, txt, conf in results:
        t = txt.strip().replace(' ', '')
        if conf < conf_thresh:
            continue  # 置信度过滤
        # 面值判断
        if t in DENOMS:
            denom = t
        # code 判断
        elif is_code(t):
            code = t
    if denom and code:
        return denom, code
    return None, None


# 识别图片内文字
def recognize_chip(img, step=15):
    if img is None:
        raise ValueError("图片读取失败")
    for angle in range(0, 181, step):
        # print(f"-------------角度{angle}-----------------")
        rotated = rotate_image(img, -angle)
        denom, code = find_value_and_code(rotated)
        if denom and code:
            return {
                'angle': angle,
                'denomination': denom,
                'code': code
            }
    return None


# 批量识别文字
def recognize_chip_batch(img, detections):
    crops = []
    for det in detections:
        crops.append(process_image(ensure_cv2_image(img), det['bbox']))

    # 4. 批量 OCR（极速）
    texts = reader.recognize(
        crops,
        batch_size=8,
        detail=0,
        paragraph=False
    )

    # 5. 对应结果
    print('---------------------------------')
    print(texts)
    print('---------------------------------')


# PIL Image / ndarray -> OpenCV ndarray (BGR)
def ensure_cv2_image(img):
    if isinstance(img, np.ndarray):
        return img
    # PIL Image
    img = np.array(img)
    # RGB -> BGR（非常重要）
    if img.ndim == 3 and img.shape[2] == 3:
        img = img[:, :, ::-1]
    return img


# 裁剪图片
def crop_with_padding(img, x1, y1, x2, y2, pad_ratio=0.15):
    h, w = img.shape[:2]

    bw = x2 - x1
    bh = y2 - y1

    pad_w = int(bw * pad_ratio)
    pad_h = int(bh * pad_ratio)

    nx1 = max(0, int(x1) - pad_w)
    ny1 = max(0, int(y1) - pad_h)
    nx2 = min(w, int(x2) + pad_w)
    ny2 = min(h, int(y2) + pad_h)

    roi = img[ny1:ny2, nx1:nx2]
    roi = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)  # 避免 EasyOCR 内部转换
    return roi


# 处理筹码图片
def process_image(img, bbox):
    if img is None:
        raise ValueError("原图读取失败")
    return crop_with_padding(
        img,
        bbox["x1"], bbox["y1"],
        bbox["x2"], bbox["y2"]
    )

# result = recognize_chip("t27.png")
#
# if result:
#     print("识别成功：", result)
# else:
#     print("识别失败")
