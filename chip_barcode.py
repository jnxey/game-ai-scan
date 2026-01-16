import cv2
from pyzbar.pyzbar import decode
# pip install opencv-python pyzbar pillow

# 读取图片
img = cv2.imread("barcode1.jpg")

# 解码
barcodes = decode(img)

for barcode in barcodes:
    data = barcode.data.decode("utf-8")
    barcode_type = barcode.type

    print("内容:", data)
    print("类型:", barcode_type)

    # 画框
    x, y, w, h = barcode.rect
    cv2.rectangle(img, (x, y), (x+w, y+h), (0,255,0), 2)

cv2.imshow("result", img)
cv2.waitKey(0)
