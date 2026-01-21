import easyocr
import cv2

# 初始化 OCR（建议全局只初始化一次）
reader = easyocr.Reader(['en'], gpu=True, verbose=False)

def preprocess_for_ocr(roi):
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    _, th = cv2.threshold(gray, 0, 255,
                          cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return th

def easyocr_digits_only(img):
    results = reader.readtext(img, detail=1, paragraph=False)
    digits = []
    for _, text, conf in results:
        if conf > 0.8 and len(text) == 6:
            digits.append(text)
    if not digits:
        print("⚠️ EasyOCR 未识别到数字")
        return None
    return digits[0]

# img = cv2.imread("easy_ocr2.png")
# result = easyocr_digits_only(img)
# print("最终结果：", result)
