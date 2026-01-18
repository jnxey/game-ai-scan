import cv2
import numpy as np
import math

def crop_text_no_finder(img):
    """
    自动裁切文字区域（不含定位框），稳定处理任意角度
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, th = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    contours, hierarchy = cv2.findContours(th, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    hierarchy = hierarchy[0]

    # 找左右定位框（外环+内环）
    finders = []
    for i, c in enumerate(contours):
        x, y, w, h = cv2.boundingRect(c)
        if w*h < 800 or w/h < 0.85 or w/h > 1.2:
            continue
        c1 = hierarchy[i][2]
        if c1 == -1: continue
        c2 = hierarchy[c1][2]
        if c2 == -1: continue
        x2, y2, w2, h2 = cv2.boundingRect(contours[c2])
        cx0, cy0 = x + w/2, y + h/2
        cx2, cy2 = x2 + w2/2, y2 + h2/2
        if abs(cx0 - cx2) > w*0.1 or abs(cy0 - cy2) > h*0.1:
            continue
        finders.append({
            "inner": (x2, y2, w2, h2),
            "center": (cx0, cy0)
        })

    if len(finders) < 2:
        raise RuntimeError("未检测到足够 Finder Pattern")

    left, right = sorted(finders, key=lambda f: f["center"][0])

    # 文字旋转角度
    lx, ly = left["center"]
    rx, ry = right["center"]
    angle = math.degrees(math.atan2(ry - ly, rx - lx))

    # -----------------------------
    # 旋转整张图，使文字水平
    # -----------------------------
    cx = (lx + rx)/2
    cy = (ly + ry)/2
    M = cv2.getRotationMatrix2D((cx, cy), angle, 1.0)
    rotated = cv2.warpAffine(img, M, (img.shape[1], img.shape[0]))

    # -----------------------------
    # 在水平图里裁切文字
    # -----------------------------
    # 左右边界：用内框中心 ± 内框宽/2
    x1 = int(left["center"][0] - left["inner"][2]/2)
    x2 = int(right["center"][0] + right["inner"][2]/2)

    # 上下边界：用内框高度 * 1.5
    h = max(left["inner"][3], right["inner"][3])
    y1 = int(cy - h * 0.75)
    y2 = int(cy + h * 0.75)

    cropped_text = rotated[y1:y2, x1:x2].copy()

    return cropped_text, 0  # 已经水平，角度返回0


img = cv2.imread("ocr_mark2.png")
roi, angle = crop_text_no_finder(img)

print("文字旋转角度已水平：", angle)
cv2.imshow("Cropped Text Only", roi)
cv2.waitKey(5000)
cv2.destroyAllWindows()
