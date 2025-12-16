import math
from itertools import combinations

poker_class = ["10C", "10D", "10H", "10S", "2C", "2D", "2H", "2S", "3C", "3D", "3H", "3S", "4C", "4D", "4H", "4S", "5C",
               "5D", "5H", "5S", "6C", "6D", "6H", "6S", "7C", "7D", "7H", "7S", "8C", "8D", "8H", "8S", "9C", "9D",
               "9H", "9S", "AC", "AD", "AH", "AS", "JC", "JD", "JH", "JS", "KC", "KD", "KH", "KS", "QC", "QD", "QH",
               "QS"]

majiang_class = ['circle_9', 'bamboo_2', 'bamboo_3', 'bamboo_4', 'bamboo_5', 'bamboo_6', 'bamboo_8', 'bamboo_9',
                 'circle_8', 'circle_7', 'circle_4', 'circle_2', 'circle_1', 'circle_3', 'circle_5', 'bamboo_7',
                 'circle_6', 'bamboo_1', 'character_9', 'character_8', 'character_7', 'character_6', 'character_5',
                 'character_3', 'character_2', 'character_4', 'green', 'red', 'north', 'east', 'character_1', 'white',
                 'west', 'south']


def yolo_to_corner(det):
    x1, y1, x2, y2 = det[:4]
    return {
        "x1": x1,
        "x2": x2,
        "y1": y1,
        "y2": y2,
        "cx": (x1 + x2) / 2,
        "cy": (y1 + y2) / 2,
        "w": abs(x2 - x1),
        "h": abs(y2 - y1),
        "angle": (abs(x2 - x1) / abs(y2 - y1)) > 1 if 1 else 0
    }


def belong_to_same_card(c1, c2):
    # 角度差不能太大
    if (c1["angle"] != c2["angle"]):
        return False

    # 中心点
    x1, y1 = c1["cx"], c1["cy"]
    x2, y2 = c2["cx"], c2["cy"]

    # 尺寸
    w = max(c1["w"], c2["w"])
    h = max(c1["h"], c2["h"])

    # 间隔太小
    if abs(x1 - x2) < w or abs(y1 - y2) < h:
        return False

    # 间隔太大
    if abs(x1 - x2) > 10 * w or abs(y1 - y2) > 10 * h:
        return False

    return True


# detections: 格式化数据
def format_poker_detections(results):
    singles = []
    pairs = []
    for r in results:
        boxes = r.boxes.xyxy.tolist()  # [[x1, y1, x2, y2], ...]
        scores = r.boxes.conf.tolist()  # 置信度
        classes = r.boxes.cls.tolist()  # 类别索引
        for b, s, c in zip(boxes, scores, classes):
            matched = False
            cBox = {
                "box": yolo_to_corner(b),
                "score": s,
                "class_name": poker_class[int(c)]
            }
            for sBox in singles:
                if sBox["class_name"] == cBox["class_name"]:
                    if belong_to_same_card(sBox["box"], cBox["box"]):
                        pairs.append({
                            "bbox": [sBox["box"], cBox["box"]],
                            "score": cBox["score"],
                            "class_name": cBox["class_name"],
                        })
                        singles.remove(sBox)
                        matched = True
                        break
            if not matched:
                singles.append(cBox)
    return pairs


def format_majiang_detections(results):
    detections = []
    for r in results:
        boxes = r.boxes.xyxy.tolist()  # [[x1, y1, x2, y2], ...]
        scores = r.boxes.conf.tolist()  # 置信度
        classes = r.boxes.cls.tolist()  # 类别索引
        for b, s, c in zip(boxes, scores, classes):
            detections.append({
                "bbox": [yolo_to_corner(b)],
                "score": s,
                "class_name": majiang_class[int(c)]
            })
    return detections
