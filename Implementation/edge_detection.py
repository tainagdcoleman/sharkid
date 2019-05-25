import cv2
import numpy as np 
from pathlib import Path
import math
from functools import partial
import matplotlib.pyplot as plt

import argparse
#comment here

from typing import Tuple, Union, List, TypeVar, Optional
Num = TypeVar('Num', int, float)

def magnitude(point: Tuple[Num, Num]) -> Num:
    return math.sqrt(point[0]**2 + point[1]**2)

def angle(point: Tuple[Num, Num], point0: Tuple[Num, Num], point1: Tuple[Num, Num]) -> float:
    """Calculates angles between three points

    Args:
        point: midpoint
        point0: first endpoint
        point1: second endpoint

    Returns:
        angle between three points in radians
    """
    a = (point[0] - point0[0], point[1] - point0[1])
    b = (point[0] - point1[0], point[1] - point1[1])

    adotb = (a[0] * b[0] + a[1] * b[1])

    return math.acos(adotb / (magnitude(a) * magnitude(b)))

def find_points(img: 'cv2.Mat', 
                kernel: int = 9, 
                thresh: int = 50) -> Tuple:
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (kernel, kernel), 0)
    _, binary = cv2.threshold(blur, thresh, 255, cv2.THRESH_BINARY_INV)

    contours, _ = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    print(contours)
    contour = contours[np.argmax(list(map(len, contours)))]

    _, tri_points = cv2.minEnclosingTriangle(contour)

    point_idxs = []
    dist = lambda p1, p2: math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)
    for tri_point in tri_points:         
        idx = np.argmin(list(map(lambda point: dist(point[0], tri_point[0]), contour)))
        point_idxs.append(idx)
    
    return sorted(point_idxs), contour, tri_points[np.argsort(point_idxs)]

def find_area(points: List[Tuple[int, int]])-> float:
    start_x, start_y = points[0] 
    end_x, end_y = points[-1]

    inner = points[1:-1]

    dy = end_y - start_y
    dx = end_x - start_x
    m_dif = end_x*start_y - end_y*start_x
    denom = math.sqrt(dy**2 + dx**2)
    
    dist = lambda point: abs(dy*point[0] - dx*point[1] + m_dif)/denom

    return sum(map(dist, inner))

def find_dist_tri(endpoints, points: List[Tuple[int, int]])-> float:
    start_x, start_y = endpoints[0]
    end_x, end_y = endpoints[1]
    mid_x, mid_y = points[len(points) // 2]
    
    dy = end_y - start_y
    dx = end_x - start_x
    m_dif = end_x*start_y - end_y*start_x
    denom = math.sqrt(dy**2 + dx**2)
    dist = abs(dy*mid_x - dx*mid_y + m_dif) / denom
    return dist

def find_angles(points: List[Tuple[int, int]])-> float:
    return -angle(points[len(points) // 2], points[0], points[-1])

def draw(points: List, 
         contour: List, 
         triangle: Optional[List] = None) -> None:

    drawing = np.full((img.shape[0], img.shape[1], 3), 0, np.uint8)
    cv2.drawContours(drawing, [contour], 0, (0, 255, 0), thickness=5)
    
    for point in points:
        x, y = point
        cv2.circle(drawing, (x, y), color=(0, 0, 255), radius=20, thickness=-1)

    if triangle is not None:
        for i, point in enumerate(triangle):
            x, y = point[0]
            x1, y1 = triangle[(i+1)%len(triangle)][0]
            cv2.line(drawing, (x, y), (x1, y1), color=(0, 255, 255), thickness=5)

    cv2.imshow('fin', drawing)
    
    if cv2.waitKey(0) == ord('n'):
        return

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--debug',
        type=bool,
        default=False,
        help='Set flag',
    ) 
    parser.add_argument(
        '--no-draw',
        type=bool,
        default=False,
        help='Do not draw',
    )

    args = parser.parse_args()

    do_draw = not args.no_draw
    do_debug = args.debug

    file_path = Path(__file__).parent.absolute()
    masks_path = Path(file_path, "Masks")

    cv2.namedWindow(f'fin', cv2.WINDOW_NORMAL)
    for mask_path in masks_path.glob('*.png'):
        img = cv2.imread(str(mask_path))
        point_idxs, contour, triangle = find_points(img)

        partitions = []

        contour_list = [tuple(elem[0]) for elem in contour]

        for i in range(len(point_idxs)):
            idx = point_idxs[i]
            idx_next = point_idxs[(i+1) % len(point_idxs)]

            if idx <= idx_next:
                partitions.append(contour_list[idx:idx_next])
            else:
                partitions.append(contour_list[idx:] + contour_list[:idx_next])


        tri_lines = ((tuple(triangle[i][0]), tuple(triangle[(i+1)%len(triangle)][0]))
                     for i in range(len(triangle)))

        methods = [
            map(find_area, partitions),
            map(find_angles, partitions),
            map(lambda args: find_dist_tri(*args), zip(tri_lines, partitions)),
        ]
        idx_max = np.argmax(list(methods[2]))
        detected_points = [partitions[idx_max][0], partitions[idx_max][-1]]

        if do_draw:
            if do_debug:
                detected_points += [p[len(p) // 2] for p in partitions]
                draw(detected_points, contour, triangle)
            else:
                draw(detected_points, contour)

        