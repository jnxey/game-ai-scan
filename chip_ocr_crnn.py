import cv2

# 加载 ONNX 模型
net = cv2.dnn.readNet("text_recognition_CRNN_EN_2023feb_fp16.onnx")

# 预处理
img = cv2.imread("easy_ocr3.png", cv2.IMREAD_GRAYSCALE)
img = cv2.resize(img, (100, 32))      # 高 32 为常见 CRNN 输入
blob = cv2.dnn.blobFromImage(img, 1/255.0, (100,32), (0,0,0), swapRB=False)

net.setInput(blob)
pred = net.forward()                  # [seq_len, batch=1, classes]

# 简单 greedy 解码 → 得到字符序列
chars = "0123456789"
pred = pred.squeeze(1)                # 去 batch
text = ""
last = -1
for t in pred:
    idx = t.argmax()
    if idx!=last and idx < len(chars):
        text += chars[idx]
    last = idx

print("识别结果:", text)
