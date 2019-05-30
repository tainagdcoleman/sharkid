import cv2
import numpy as np 
import pathlib
import math 
import matplotlib.pyplot as plt 
import networkx as nx
import itertools

from typing import Tuple, List
from helpers import Num, Point, dist_to_line, gaussian_filter, coord_transform, remove_decreasing, distance
from scipy.signal import find_peaks, peak_prominences


class Fin:
    def __init__(self, path_image:str) -> None:
        self.path_image = pathlib.Path(path_image)        
        if not self.path_image.is_file():
            raise FileNotFoundError(self.path_image)

        self.contour = None 
        self.point_idxs = None
        self.triangle = None

        self.draw_points = []

        self.points = []
        self.peaks = []
        self.prominences = []

        self.is_processed = False
        try:
            self._process_image()
            self.is_processed = True
        except Exception as e:
            print(e)

    def _process_image(self) -> None:
        self.img = cv2.imread(str(self.path_image)) #load image
        self.point_idxs, self.contour, self.triangle = self._find_contour(self.img) 

        # treat the self.contour to return a simple list of Tuples
        contour_list = [tuple(elem[0]) for elem in self.contour] 
        
        # partition contour into 3 segments
        partitions = []
        for i in range(len(self.point_idxs)):
            idx = self.point_idxs[i]
            idx_next = self.point_idxs[(i+1) % len(self.point_idxs)]

            if idx <= idx_next:
                partitions.append(contour_list[idx:idx_next])
            else:
                partitions.append(contour_list[idx:] + contour_list[:idx_next])

        # find the partitions that contains the ridges by 
        # finding the middle point of each partition and measuring 
        # their distance to the bounding triangle
        max_dist = -1
        ridges = None
        for i in range(len(self.triangle)):
            start = self.triangle[i][0]
            end = self.triangle[(i+1)%len(self.triangle)][0]
            mid = partitions[i][len(partitions[i]) // 2]
            dist = dist_to_line(start, end, mid)
            if max_dist < dist:
                ridges = partitions[i]
                max_dist = dist
        
        if max_dist <= 0:
            raise Exception(f'Could not find correct partition for {self.path_image}')

        self.draw_points += [ridges[0], ridges[-1]]

        # return the ridges as normalized 2-dim signal 
        xs, ys = coord_transform(ridges)

        # Get rid of bad points (x should be increasing)
        xs, ys = map(np.asarray, zip(*remove_decreasing(xs, ys)))

        # normalize 
        xs = xs / np.absolute(xs).max()
        ys = ys / np.absolute(ys).max()

        # apply gaussian filter to the ridges partition
        self.points = gaussian_filter(list(zip(xs, ys)))

        # find peak prominences
        _, ys = zip(*self.points)
        self.peaks, _ = find_peaks(ys)
        self.prominences = peak_prominences(ys, self.peaks)[0]
        
    def _find_contour(self,
                      img: 'cv2.Mat',
                      kernel: int = 9, 
                      thresh: int = 50) -> Tuple:
        
        # preprocessing the image
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (kernel, kernel), 0)
        _, binary = cv2.threshold(blur, thresh, 255, cv2.THRESH_BINARY_INV)

        # getting contour
        contours, _ = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        contour = contours[np.argmax(list(map(len, contours)))] # exclude noise contours 

        _, tri_points = cv2.minEnclosingTriangle(contour)

        # for each of the triangle's vertices, find the closest point in the contour     
        point_idxs = []
        dist = lambda p1, p2: math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)
        for tri_point in tri_points:         
            idx = np.argmin(list(map(lambda point: dist(point[0], tri_point[0]), contour)))
            point_idxs.append(idx)
        
        return sorted(point_idxs), contour, tri_points[np.argsort(point_idxs)]

    def _get_ridges(self, points: List[Point]) -> None:
        xs, ys = coord_transform(points)

        # Get rid of bad points (x should be increasing)
        xs, ys = map(np.asarray, zip(*remove_decreasing(xs, ys)))

        # normalize 
        xs = xs / np.absolute(xs).max()
        ys = ys / np.absolute(ys).max()

        return xs, ys        
    
    def draw(self, draw_triangle:bool = False) -> 'cv2.Mat':
        if not self.is_processed:
            raise Exception(f'Cannot draw {self.path_image} since it could not be processed.')

        drawing = np.full((self.img.shape[0], self.img.shape[1], 3), 0, np.uint8)
        cv2.drawContours(drawing, [self.contour], 0, (0, 255, 0), 
                         thickness=5)
        
        for x, y in self.draw_points:
            cv2.circle(drawing, (x, y), color=(0, 0, 255), 
                       radius=20, thickness=-1)

        if draw_triangle:
            for i, point in enumerate(self.triangle):
                x, y = point[0]
                x1, y1 = self.triangle[(i+1)%len(self.triangle)][0]
                cv2.line(drawing, (x, y), (x1, y1), 
                         color=(0, 255, 255), thickness=5)

        return drawing 

    def plot(self, with_image: bool = False) -> 'plt.Figure':
        if not self.is_processed:
            raise Exception(f'Cannot plot {self.path_image} since it could not be processed.')

        if with_image:
            fig, (ax_im, ax) = plt.subplots(2, 1)
            ax_im.imshow(self.draw())
        else:
            fig, ax = plt.subplots(1)

        ax.plot(*zip(*self.points))
        ax.plot(*zip(*self.points[self.peaks]), 'ro')

        return fig

    def save(self, save_path: str) -> None:   
        if not self.is_processed:
            raise Exception(f'Cannot save {self.path_image} since it could not be processed.')

        save_path = pathlib.Path(save_path)
        if save_path.is_dir():
            raise FileExistsError(f'{save_path} is a directory.')
        save_path.parent.mkdir(exist_ok=True, parents=True)
        fig = self.plot(with_image=True)
        fig.savefig(str(save_path))

    def similarity(self, other: 'Fin', maxn: int = 50) -> float:
        if not isinstance(other, Fin):
            raise ValueError(f'{other} is not a Fin.')
        
        maxn = min([len(self.peaks), len(other.peaks), maxn])
        self_idxs = np.argpartition(self.prominences, -maxn)[-maxn:]
        other_idxs = np.argpartition(other.prominences, -maxn)[-maxn:]

        g = nx.Graph()
        
        self_pts = list(enumerate(self.points[self.peaks]))
        other_pts = list(enumerate(other.points[other.peaks]))
        for (i, point1), (j, point2) in itertools.product(self_pts, other_pts):
            g.add_edge(('s', i), ('d', j), weight=-distance(point1, point2))

        matching = nx.bipartite.maximum_matching(g)
        score = sum([g.get_edge_data(k, v)['weight'] for k, v in matching.items()])

        return score