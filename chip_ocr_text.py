import cv2
import numpy as np
import random
import math

DEBUG = False

def dbg(name, img, wait=0):
    if DEBUG:
        cv2.imshow(name, img)
        cv2.waitKey(wait)

# =========================
# 统一宽度
# =========================
def resize_to_width(img, target_w=640):
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
        if 10 <= w <= box_max_size and 10 <= h <= box_max_size:
            candidates.append((x, y, w, h))
    return candidates

# =========================
# RANSAC 找连续直线点（纯 numpy）
# =========================
def ransac_linear_cluster_numpy(img, centers, min_count=5, line_tol_px=5, max_trials=500):
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

    group_filtered = filter_points_by_distance(best_group)

    # debug 可视化
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
def filter_points_by_distance(points, min_dist=45, max_dist=135):
    if len(points) < 2:
        return points
    filtered = [points[0]]
    for pt in points[1:]:
        last = filtered[-1]
        dist = np.linalg.norm(pt - last)
        if min_dist <= dist <= max_dist:
            filtered.append(pt)
        elif dist > max_dist:
            break
    return np.array(filtered)

# =========================
# 旋转裁切 ROI
# =========================
def rotate_and_crop_roi(img, points, box_max_size):
    if len(points) < 2:
        return None, 0

    # =========================
    # 1. 计算中心 & 角度
    # =========================
    cx = np.mean(points[:, 0])
    cy = np.mean(points[:, 1])

    dx, dy = points[-1] - points[0]
    angle = math.degrees(math.atan2(dy, dx))

    h_img, w_img = img.shape[:2]

    # =========================
    # 2. 按文字方向旋转
    # =========================
    M = cv2.getRotationMatrix2D((cx, cy), angle, 1.0)
    rotated = cv2.warpAffine(img, M, (w_img, h_img))
    dbg("rotated", rotated)

    # =========================
    # 3. 裁剪 ROI
    # =========================
    avg_w = box_max_size * 1.2  # 可后续用真实文字高度替换
    avg_h = box_max_size / 1.2  # 可后续用真实文字高度替换

    x1 = int(np.min(points[:, 0]) - avg_w)
    x2 = int(np.max(points[:, 0]) + avg_w)

    y_center = np.mean(points[:, 1])
    y1 = int(y_center - avg_h)
    y2 = int(y_center + avg_h)

    roi = rotated[y1:y2, x1:x2]
    dbg("roi_before_fix", roi)

    # =========================
    # 4. ⭐ 核心新增逻辑：判断是否需要 180° 翻转
    # =========================
    if y_center < h_img / 2.5:
        # 位于图片偏上 → 翻转 180°
        roi = cv2.rotate(roi, cv2.ROTATE_180)
        angle = (angle + 180) % 360
        dbg("roi_after_180", roi)

    return roi, angle


# =========================
# 总入口 大小均指640下的大小
# =========================
def detect_and_crop(img, box_max_size=60):
    img, scale = resize_to_width(img, 640)
    bin_img = preprocess(img)

    blocks = find_text_blocks(bin_img, box_max_size)
    dbg_img = img.copy()
    for x, y, w, h in blocks:
        cv2.rectangle(dbg_img, (x, y), (x + w, y + h), (0,255,0), 1)
    dbg("blocks", dbg_img)

    centers = [(x+w/2, y+h/2) for x,y,w,h in blocks]
    group = ransac_linear_cluster_numpy(img, centers)
    if group is None:
        raise RuntimeError("未检测到连续字符集合")
    roi, angle = rotate_and_crop_roi(img, group, box_max_size)
    return roi, angle

# =========================
# 运行示例
# =========================
# if __name__ == "__main__":
#     img = cv2.imread("ocr_mark9.png")
#     roi, angle = detect_and_crop(img)
#     print("旋转角度:", angle)
#     cv2.waitKey(0)
#     cv2.destroyAllWindows()
