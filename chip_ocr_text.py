import cv2
import numpy as np
import random
import math
import time

DEBUG = False
DEBUG_PR = True

def dbg(name, img, wait=0):
    if DEBUG:
        cv2.imshow(name, img)
        cv2.waitKey(wait)

def dpr(name, value):
    if DEBUG_PR:
        print(name, value)

# =========================
# 统一宽度
# =========================
def resize_to_width(img, target_w):
    h, w = img.shape[:2]
    scale = target_w / w
    new_h = int(h * scale)
    resized = cv2.resize(img, (target_w, new_h))
    return resized, scale

# =========================
# 预处理：灰度 + 二值
# =========================
def preprocess(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    dbg("1_gray", gray)

    bin_img = cv2.adaptiveThreshold(
        gray, 255,
        cv2.ADAPTIVE_THRESH_MEAN_C,
        cv2.THRESH_BINARY_INV,
        31, 5
    )
    dbg("2_binary", bin_img)
    return bin_img

# =========================
# 找文字块
# =========================
def find_text_blocks(bin_img, box_max_size):
    contours, _ = cv2.findContours(bin_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    candidates = []
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        min_size = box_max_size * 0.2
        if(w <= box_max_size * 0.6 and h <= box_max_size * 0.6):
            continue
        if min_size <= w <= box_max_size and min_size <= h <= box_max_size:
            candidates.append((x, y, w, h))
    return candidates

# =========================
# RANSAC 找连续直线点（纯 numpy）
# =========================
def ransac_linear_cluster_numpy(img, centers, box_max_size, min_count, line_tol_px, max_trials):
    centers = np.array(centers, dtype=np.float32)
    n = len(centers)
    if n < min_count:
        return None

    best_group = []

    for _ in range(max_trials):
        p1, p2 = centers[random.sample(range(n), 2)]
        if np.all(p1 == p2):
            continue
        line_vec = p2 - p1
        line_len = np.linalg.norm(line_vec)
        if line_len < 1e-5:
            continue
        line_dir = line_vec / line_len

        diffs = centers - p1
        proj_lengths = np.dot(diffs, line_dir)
        projections = p1 + np.outer(proj_lengths, line_dir)
        dists = np.linalg.norm(centers - projections, axis=1)

        inliers = centers[dists <= line_tol_px]

        if len(inliers) >= min_count and len(inliers) > len(best_group):
            inliers = inliers[np.argsort(inliers[:,0])]
            best_group = inliers

    if len(best_group) < min_count:
        return None

    group_filtered = find_compact_x_cluster(best_group, box_max_size * 2.5)

    # debug 可视化
    if DEBUG:
        dbg_img = img.copy()
        for pt in centers:
            cv2.circle(dbg_img, (int(pt[0]), int(pt[1])), 3, (0,255,0), -1)
        for pt in group_filtered:
            cv2.circle(dbg_img, (int(pt[0]), int(pt[1])), 5, (0,0,255), -1)
        if len(group_filtered) >= 2:
            cv2.line(dbg_img, tuple(group_filtered[0].astype(int)), tuple(group_filtered[-1].astype(int)), (255,0,0), 1)
        dbg("ransac_cluster", dbg_img)

    return group_filtered

# =========================
# 筛选相邻点距离
# =========================
def find_compact_x_cluster(points, max_gap, min_count=2):
    """
    points: Nx2 (只用 x)
    max_gap: 允许的最大相邻 x 距离
    min_count: 至少多少个点才算有效
    """
    if len(points) < min_count:
        return None

    pts = np.array(points, dtype=np.float32)

    # 1️⃣ 按 x 排序
    order = np.argsort(pts[:, 0])
    pts = pts[order]

    best_cluster = []
    curr_cluster = [pts[0]]

    for i in range(1, len(pts)):
        dx = pts[i][0] - pts[i - 1][0]

        if dx <= max_gap:
            curr_cluster.append(pts[i])
        else:
            # 断开，比较
            if len(curr_cluster) > len(best_cluster):
                best_cluster = curr_cluster.copy()
            curr_cluster = [pts[i]]

    # 最后一段
    if len(curr_cluster) > len(best_cluster):
        best_cluster = curr_cluster.copy()

    if len(best_cluster) < min_count:
        return None

    return np.array(best_cluster)



# =========================
# 旋转裁切 ROI
# =========================
def rotate_and_crop_roi_image_center(
        img,
        points,
        box_max_size,
        pad_x_ratio=1.5,
        pad_y_ratio=1.2,
        border_mode=cv2.BORDER_REPLICATE
):
    """
    ✅ 正确版：以【图像中心】旋转，而不是文字中心
    """

    if points is None or len(points) < 2:
        return None, 0

    points = np.asarray(points, dtype=np.float32)

    h, w = img.shape[:2]
    img_center = (w / 2.0, h / 2.0)

    # =========================
    # 1. 文字方向角
    # =========================
    dx, dy = points[-1] - points[0]
    angle = math.degrees(math.atan2(dy, dx))

    # =========================
    # 2. 以「图像中心」旋转整图
    # =========================
    M = cv2.getRotationMatrix2D(img_center, angle, 1.0)

    rotated = cv2.warpAffine(
        img,
        M,
        (w, h),
        flags=cv2.INTER_LINEAR,
        borderMode=border_mode
    )

    dbg("rotated", rotated)

    # =========================
    # 3. 同步旋转 points
    # =========================
    ones = np.ones((len(points), 1), dtype=np.float32)
    pts_h = np.hstack([points, ones])   # Nx3
    rot_pts = pts_h @ M.T               # Nx2

    # =========================
    # 4. 基于旋转后的点裁 ROI
    # =========================
    pad_x = box_max_size * pad_x_ratio
    pad_y = box_max_size * pad_y_ratio

    x1 = int(np.min(rot_pts[:, 0]) - pad_x)
    x2 = int(np.max(rot_pts[:, 0]) + pad_x)

    y_center = np.mean(rot_pts[:, 1])
    y1 = int(y_center - pad_y)
    y2 = int(y_center + pad_y)

    # clamp
    x1 = max(0, x1)
    y1 = max(0, y1)
    x2 = min(w, x2)
    y2 = min(h, y2)

    if x2 <= x1 or y2 <= y1:
        return None, angle

    roi = rotated[y1:y2, x1:x2]

    dbg("roi_before_fix1", roi)

    # =========================
    # 5. 可选：统一方向（可关）
    # =========================
    # print(y_center,'--------------------y_center')
    # print(h / 2,'--------------------h ')
    if y_center < (h / 2):
        roi = cv2.rotate(roi, cv2.ROTATE_180)
        angle = (angle + 180) % 360

    dbg("roi_before_fix2", roi)

    return roi, angle




# =========================
# 总入口 大小均指640下的大小
# =========================
def detect_and_crop(img, box_max_size=40):
    dpr("--------------------time-------------------------", 0)
    t1 = time.time()
    img, scale = resize_to_width(img, 640)
    bin_img = preprocess(img)
    t2 = time.time()
    dpr("灰度耗时:", t2 - t1)
    blocks = find_text_blocks(bin_img, box_max_size)
    t3 = time.time()
    dpr("查找块耗时:", t3 - t2)
    if DEBUG:
        dbg_img = img.copy()
        for x, y, w, h in blocks:
            cv2.rectangle(dbg_img, (x, y), (x + w, y + h), (0,255,0), 1)
        dbg("blocks", dbg_img)
    t4 = time.time()
    dpr("debug耗时:", t4 - t3)
    centers = [(x+w/2, y+h/2) for x,y,w,h in blocks]
    group = ransac_linear_cluster_numpy(img, centers, box_max_size, min_count=5, line_tol_px=4, max_trials=100)
    t5 = time.time()
    dpr("group耗时:", t5 - t4)
    if group is None:
        raise RuntimeError("未检测到连续字符集合")
    roi, angle = rotate_and_crop_roi_image_center(img, group, box_max_size)
    t6 = time.time()
    dpr("裁切耗时:", t6 - t5)
    dpr("总耗时:", t6 - t1)
    return roi, angle

# =========================
# 运行示例
# =========================
# if __name__ == "__main__":
#     img = cv2.imread("ocr_mark1.png")
#     roi, angle = detect_and_crop(img)
#     print("旋转角度:", angle)
#     cv2.waitKey(0)
#     cv2.destroyAllWindows()
