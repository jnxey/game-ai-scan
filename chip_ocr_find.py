import cv2
import numpy as np
import math

def show(name, img, wait=0):
    cv2.imshow(name, img)
    cv2.waitKey(wait)


# =========================
# 工具：统一宽度
# =========================
def resize_to_width(img, target_w=640):
    h, w = img.shape[:2]
    scale = target_w / w
    new_h = int(h * scale)
    resized = cv2.resize(img, (target_w, new_h))
    return resized, scale

def crop_text_no_finder_debug(img):
    # 0. 统一尺寸
    img, scale = resize_to_width(img, 640)

    # =============================
    # 1. 灰度
    # =============================
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    show("01_gray", gray)

    # =============================
    # 2. OTSU 二值化
    # =============================
    th = cv2.adaptiveThreshold(
        gray, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        11, 2
    )
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
    th_clean = cv2.morphologyEx(th, cv2.MORPH_OPEN, kernel)
    show("02_binary_otsu", th)

    # =============================
    # 3. 查找轮廓（TREE）
    # =============================
    contours, hierarchy = cv2.findContours(
        th, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
    )
    hierarchy = hierarchy[0]

    # 所有轮廓可视化
    all_cnt_img = img.copy()
    cv2.drawContours(all_cnt_img, contours, -1, (0, 255, 0), 1)
    show("03_all_contours", all_cnt_img)

    # =============================
    # 4. Finder Pattern 识别
    # =============================
    finders = []
    candidate_img = img.copy()

    for i, c in enumerate(contours):
        x, y, w, h = cv2.boundingRect(c)

        # ---- 面积 & 正方形过滤 ----
        if w * h < 1600 or w * h > 10000 or w / h < 0.85 or w / h > 1.2:
            continue

        cv2.rectangle(candidate_img, (x, y), (x + w, y + h), (255, 0, 0), 1)

        # ---- 层级检查 ----
        c1 = hierarchy[i][2]
        if c1 == -1:
            continue
        c2 = hierarchy[c1][2]
        if c2 == -1:
            continue

        x2, y2, w2, h2 = cv2.boundingRect(contours[c2])

        # ---- 同心校验 ----
        cx0, cy0 = x + w / 2, y + h / 2
        cx2, cy2 = x2 + w2 / 2, y2 + h2 / 2

        if abs(cx0 - cx2) > w * 0.1 or abs(cy0 - cy2) > h * 0.1:
            continue

        finders.append({
            "outer": (x, y, w, h),
            "inner": (x2, y2, w2, h2),
            "center": (cx0, cy0)
        })

    show("04_candidate_squares", candidate_img)

    # =============================
    # 5. Finder 可视化
    # =============================
    finder_img = img.copy()
    for f in finders:
        x, y, w, h = f["outer"]
        x2, y2, w2, h2 = f["inner"]
        cx, cy = map(int, f["center"])

        cv2.rectangle(finder_img, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.rectangle(finder_img, (x2, y2), (x2 + w2, y2 + h2), (0, 0, 255), 2)
        cv2.circle(finder_img, (cx, cy), 4, (255, 0, 0), -1)

    show("05_finder_patterns", finder_img)

    if len(finders) < 2:
        raise RuntimeError("未检测到足够 Finder Pattern")

    # =============================
    # 6. 旋转校正
    # =============================
    finders_sorted = sorted(finders, key=lambda f: f["center"][0])
    # 取最左和最右
    left = finders_sorted[0]
    right = finders_sorted[-1]
    lx, ly = left["center"]
    rx, ry = right["center"]

    angle = math.degrees(math.atan2(ry - ly, rx - lx))
    print("检测到旋转角度:", angle)

    # =============================
    # 判断是否需要旋转 180°
    # =============================
    img_height = gray.shape[0]  # 图片高度
    # 如果左右中心点平均 y 值偏上半部分，则旋转 180°
    center_y_mean = (ly + ry) / 2
    if center_y_mean < img_height / 2:
        angle = (angle + 180) % 360
        print("上下翻转，旋转 180° 后角度:", angle)

    cx = (lx + rx) / 2
    cy = (ly + ry) / 2
    M = cv2.getRotationMatrix2D((cx, cy), angle, 1.0)
    rotated = cv2.warpAffine(img, M, (img.shape[1], img.shape[0]))

    show("06_rotated", rotated)

    # =============================
    # 7. 裁剪文字区域
    # =============================
    padding_size = 10
    x1 = int(left["center"][0] - padding_size)
    x2 = int(right["center"][0] + padding_size)

    # 上下边界：用内框高度 * 1.5
    h = max(left["inner"][3], right["inner"][3])
    y1 = int(cy - h - padding_size)
    y2 = int(cy + h + padding_size)

    cropped_text = rotated[y1:y2, x1:x2].copy()
    show("07_cropped_text", cropped_text)

    return cropped_text, 0

img = cv2.imread("ocr_mark6.png")
roi, angle = crop_text_no_finder_debug(img)

print("文字旋转角度已水平：", angle)
cv2.imshow("Cropped Text Only", roi)
cv2.waitKey(0)
