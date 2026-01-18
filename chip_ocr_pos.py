import cv2
import numpy as np
import math

def locate_text_roi_with_rotation(img):
    """
    Finder Pattern（二维码定位符）定位 + 自动旋转纠偏 + 文字 ROI 裁剪
    返回：roi_img, roi_bbox(x1,y1,x2,y2), angle_deg
    """

    H, W = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    _, th = cv2.threshold(
        gray, 0, 255,
        cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )

    contours, hierarchy = cv2.findContours(
        th, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
    )
    hierarchy = hierarchy[0]

    finders = []

    for i, c in enumerate(contours):
        x, y, w, h = cv2.boundingRect(c)
        area = w * h
        ratio = w / h if h else 0

        if area < 800:
            continue
        if ratio < 0.85 or ratio > 1.2:
            continue

        c1 = hierarchy[i][2]
        if c1 == -1:
            continue

        c2 = hierarchy[c1][2]
        if c2 == -1:
            continue

        x2, y2, w2, h2 = cv2.boundingRect(contours[c2])

        # 同心校验
        cx0, cy0 = x + w/2, y + h/2
        cx2, cy2 = x2 + w2/2, y2 + h2/2
        if abs(cx0 - cx2) > w * 0.1 or abs(cy0 - cy2) > h * 0.1:
            continue

        finders.append({
            "outer": (x, y, w, h),
            "inner": (x2, y2, w2, h2),
            "center": (cx0, cy0)
        })

    if len(finders) < 2:
        raise RuntimeError("未检测到足够的 Finder")

    # 左右 Finder
    finders = sorted(finders, key=lambda f: f["center"][0])
    left = finders[0]
    right = finders[-1]

    # === 1️⃣ 计算旋转角 ===
    (lx, ly) = left["center"]
    (rx, ry) = right["center"]

    angle = math.degrees(math.atan2(ry - ly, rx - lx))
    # angle > 0 表示顺时针倾斜

    # === 2️⃣ 旋转整图 ===
    M = cv2.getRotationMatrix2D((W // 2, H // 2), angle, 1.0)
    rotated = cv2.warpAffine(
        img, M, (W, H),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REPLICATE
    )

    # === 3️⃣ 在旋转后的图上重新裁 ROI（用原 Finder 尺寸） ===
    # 注意：Finder 尺寸旋转前后基本不变（小角度）

    lh_inner = left["inner"][3]
    rh_inner = right["inner"][3]

    roi_h = int(max(lh_inner, rh_inner) * 1.8)
    pad = int(roi_h * 0.12)
    roi_h += pad * 2

    cy = int((ly + ry) / 2)
    x1 = int(left["outer"][0] + left["outer"][2])
    x2 = int(right["outer"][0])

    y1 = cy - roi_h // 2
    y2 = cy + roi_h // 2

    # 边界保护
    x1 = max(0, x1)
    y1 = max(0, y1)
    x2 = min(W, x2)
    y2 = min(H, y2)

    roi = rotated[y1:y2, x1:x2]

    return roi, (x1, y1, x2, y2), angle


img = cv2.imread("ocr_mark3.png")
roi, bbox, angle = locate_text_roi_with_rotation(img)

print("纠偏角度:", angle)

vis = img.copy()
x1, y1, x2, y2 = bbox
cv2.rectangle(vis, (x1, y1), (x2, y2), (0,255,0), 2)

cv2.imshow("ROI", roi)
cv2.waitKey(5000)  # 自动显示 5 秒
cv2.destroyAllWindows()
