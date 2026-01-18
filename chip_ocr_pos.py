import cv2
import numpy as np
import math

def crop_text_roi_from_finder(img):
    """
    自动通过左右 Finder Pattern 定位文字区域并裁切旋转矩形
    返回：
        cropped_img - 裁切后的旋转矩形区域
        angle       - 原文字旋转角度（顺时针）
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, th = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    contours, hierarchy = cv2.findContours(th, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    hierarchy = hierarchy[0]

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
            "outer": (x, y, w, h),
            "inner": (x2, y2, w2, h2),
            "center": (cx0, cy0)
        })

    if len(finders) < 2:
        raise RuntimeError("未检测到足够 Finder Pattern")

    # 左右 Finder
    finders = sorted(finders, key=lambda f: f["center"][0])
    left, right = finders[0], finders[-1]

    # 旋转角度
    lx, ly = left["center"]
    rx, ry = right["center"]
    angle = math.degrees(math.atan2(ry - ly, rx - lx))

    # 矩形中心与大小
    cx = (lx + rx) / 2
    cy = (ly + ry) / 2
    width = math.hypot(rx - lx, ry - ly)
    height = max(left["inner"][3], right["inner"][3]) * 2
    pad = int(max(left["inner"][3], right["inner"][3]) * 0.3)
    height += pad*2

    rect = ((cx, cy), (width, height), angle)

    # -----------------------------
    # 旋转裁切
    # -----------------------------
    center, size, angle = rect
    center = tuple(map(int, center))
    size = tuple(map(int, size))
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(img, M, (img.shape[1], img.shape[0]))
    x, y = int(center[0] - size[0]/2), int(center[1] - size[1]/2)
    cropped = rotated[y:y+size[1], x:x+size[0]].copy()

    return cropped, angle

img = cv2.imread("ocr_mark3.png")
roi, angle = crop_text_roi_from_finder(img)

print("文字旋转角度：", angle)
cv2.imshow("Cropped ROI", roi)
cv2.waitKey(5000)
cv2.destroyAllWindows()
