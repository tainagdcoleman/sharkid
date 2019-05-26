import cv2
import os
import numpy as np 
import pathlib
import math 
import matplotlib.pyplot as plt 

import argparse

from typing import Tuple, Union, List, TypeVar, Optional
from helpers import Num, Point, dist_to_line

from functools import partial
from multiprocessing import Pool 

def find_points(img: 'cv2.Mat', 
                kernel: int = 9, 
                thresh: int = 50) -> Tuple:
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (kernel, kernel), 0)
    _, binary = cv2.threshold(blur, thresh, 255, cv2.THRESH_BINARY_INV)

    contours, _ = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contour = contours[np.argmax(list(map(len, contours)))]

    _, tri_points = cv2.minEnclosingTriangle(contour)

    point_idxs = []
    dist = lambda p1, p2: math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)
    for tri_point in tri_points:         
        idx = np.argmin(list(map(lambda point: dist(point[0], tri_point[0]), contour)))
        point_idxs.append(idx)
    
    return sorted(point_idxs), contour, tri_points[np.argsort(point_idxs)]
    
def coord_transform(points: List[Tuple[int, int]])-> float:
    start_x, start_y = points[0] 
    end_x, end_y = points[-1]
    inner = points[1:-1]

    perp_point_x = start_x - (end_y - start_y)
    perp_point_y = start_y + (end_x - start_x)

    ys = dist_to_line((start_x, start_y), (end_x, end_y), *inner, signed=True)
    xs = dist_to_line((start_x, start_y), (perp_point_x, perp_point_y), *inner, signed=True)

    return xs, ys

def draw(img: 'cv2.Mat',
         points: List, 
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

    return drawing 

def remove_decreasing(xs, ys):
    maxx = xs[0]
    for x, y in zip(xs, ys):
        if x > maxx:
            yield x, y
            maxx = x

def get_ridges(points: List[Point]) -> None:
    xs, ys = coord_transform(points)

    # Get rid of bad points (x should be increasing)
    xs, ys = map(np.asarray, zip(*remove_decreasing(xs, ys)))

    # normalize 
    xs = xs / np.absolute(xs).max()
    ys = ys / np.absolute(ys).max()

    return xs, ys 

def process_image(mask_path: pathlib.Path, 
                  outdir: pathlib.Path,
                  do_debug: bool = False,
                  do_draw: bool = False) -> None:
    img = cv2.imread(str(mask_path))
    point_idxs, contour, triangle = find_points(img)

    contour_list = [tuple(elem[0]) for elem in contour]
    partitions = []
    for i in range(len(point_idxs)):
        idx = point_idxs[i]
        idx_next = point_idxs[(i+1) % len(point_idxs)]

        if idx <= idx_next:
            partitions.append(contour_list[idx:idx_next])
        else:
            partitions.append(contour_list[idx:] + contour_list[:idx_next])

    max_dist = -1
    ridges = None
    for i in range(len(triangle)):
        start = triangle[i][0]
        end = triangle[(i+1)%len(triangle)][0]
        mid = partitions[i][len(partitions[i]) // 2]
        dist = dist_to_line(start, end, mid)
        if max_dist < dist:
            ridges = partitions[i]
            max_dist = dist
    
    if max_dist <= 0:
        print(f'Error with {mask_path}')
        return

    fig, (ax_im, ax_ridges) = plt.subplots(2, 1)
    if do_debug:
        detected_points = [ridges[0], ridges[-1]] + [p[len(p) // 2] for p in partitions]
        drawing = draw(img, detected_points, contour, triangle)
    else:
        drawing = draw(img, [ridges[0], ridges[-1]], contour)

    ax_im.imshow(drawing)

    xs, ys = get_ridges(ridges)
    ax_ridges.plot(xs, ys)

    if do_draw:
        fig.show()
    
    if outdir:
        fig.savefig(f'{outdir.joinpath(mask_path.stem)}.png')
    
    plt.close(fig)


if __name__ == '__main__':    
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--debug',
        action='store_true',
        help='Set debug flag for plotting extra points',
    ) 
    parser.add_argument(
        '--show',
        action='store_true',
        help='Display the output, one at a time',
    ) 
    parser.add_argument(
        '--out',
        default='out',
        help='output directory',
    )

    args = parser.parse_args()

    outdir = pathlib.Path(args.out)
    if outdir.is_file():
        raise FileExistsError(f'{outdir} is a file. You should provide a directory.')
    if not outdir.is_dir():
        outdir.mkdir(parents=True, exist_ok=True)

    file_path = pathlib.Path(__file__).parent.absolute()
    masks_path = pathlib.Path(file_path, "Masks")

    pool = Pool(os.cpu_count())
    process_func = partial(process_image, outdir=outdir, do_debug=args.debug, do_draw=args.show)
    pool.map(partial(process_image, outdir=outdir), masks_path.glob('*.png'))