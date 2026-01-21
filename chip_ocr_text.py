import cv2
import numpy as np
import random
import math

DEBUG = True

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
def find_text_blocks(bin_img):
    contours, _ = cv2.findContours(bin_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    candidates = []
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        if 10 <= w <= 150 and 10 <= h <= 150:
            candidates.append((x, y, w, h))
    return candidates

# =========================
# RANSAC 找连续直线点（纯 numpy）
# =========================
def ransac_linear_cluster_numpy(img, centers, min_count=5, max_count=16, line_tol_px=5, max_trials=500):
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
            if len(inliers) > max_count:
                inliers = inliers[:max_count]
            best_group = inliers

    if len(best_group) < min_count:
        return None

    # debug 可视化
    dbg_img = img.copy()
    for pt in centers:
        cv2.circle(dbg_img, (int(pt[0]), int(pt[1])), 3, (0,255,0), -1)
    for pt in best_group:
        cv2.circle(dbg_img, (int(pt[0]), int(pt[1])), 5, (0,0,255), -1)
    if len(best_group) >= 2:
        cv2.line(dbg_img, tuple(best_group[0].astype(int)), tuple(best_group[-1].astype(int)), (255,0,0), 1)
    dbg("ransac_cluster", dbg_img)

    return best_group

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
def rotate_and_crop_roi(img, points):
    if len(points) < 2:
        return None, 0
    cx = np.mean(points[:,0])
    cy = np.mean(points[:,1])

    dx, dy = points[-1] - points[0]
    angle = math.degrees(math.atan2(dy, dx))

    h_img, w_img = img.shape[:2]
    M = cv2.getRotationMatrix2D((cx, cy), angle, 1.0)
    rotated = cv2.warpAffine(img, M, (w_img, h_img))
    dbg("rotated", rotated)

    avg_h = np.mean([10 if h<10 else h for h in [50]*len(points)])  # 可以改成文字高度
    x1 = int(np.min(points[:,0]) - avg_h)
    x2 = int(np.max(points[:,0]) + avg_h)
    y_center = np.mean(points[:,1])
    y1 = int(y_center - avg_h)
    y2 = int(y_center + avg_h)
    roi = rotated[y1:y2, x1:x2]
    dbg("roi", roi)
    return roi, angle

# =========================
# 总入口
# =========================
def detect_and_crop(img):
    img, scale = resize_to_width(img, 640)
    bin_img = preprocess(img)

    blocks = find_text_blocks(bin_img)
    dbg_img = img.copy()
    for x, y, w, h in blocks:
        cv2.rectangle(dbg_img, (x, y), (x + w, y + h), (0,255,0), 1)
    dbg("blocks", dbg_img)

    centers = [(x+w/2, y+h/2) for x,y,w,h in blocks]
    group = ransac_linear_cluster_numpy(img, centers, min_count=5, max_count=8, line_tol_px=5)
    if group is None:
        raise RuntimeError("未检测到连续字符集合")

    group_filtered = filter_points_by_distance(group, min_dist=45, max_dist=135)
    roi, angle = rotate_and_crop_roi(img, group_filtered)
    return roi, angle

# =========================
# 运行示例
# =========================
if __name__ == "__main__":
    img = cv2.imread("ocr_mark1.png")
    roi, angle = detect_and_crop(img)
    print("旋转角度:", angle)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
