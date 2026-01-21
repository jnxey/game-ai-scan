import easyocr
import cv2

reader = easyocr.Reader(
    ['en'],  # en 就够（数字也在这里）
    gpu=True,
    verbose=False
)

img = cv2.imread("easy_ocr1.png")


def easyocr_digits_only(img):
    results = reader.readtext(img, detail=1, paragraph=False)

    digits = []
    for _, text, conf in results:
        print(conf,'-----1')
        if conf > 0.4:
            digits.append(text)

    return digits


result = easyocr_digits_only(img)
print("最终结果：", result)
