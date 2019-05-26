from typing import Tuple, List, Dict, TypeVar

Num = TypeVar('Num', float, int)
Point = Tuple[Num, Num]

def angle(point: Point, 
          point0: Point, 
          point1: Point) -> float:
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


def find_angles(points: List[Point])-> List[float]:
    """Finds angles between all points in sequence of points

    Args:
        points: sequential list of points

    Returns:
        angles in radians

    """
    return angle(points[len(points) // 2], points[0], points[-1])


def magnitude(point: Point) -> float:
    """Finds the magnitude of a point, as if it were a vector originating at 0, 0

    Args:
        point: Point (x, y)

    Returns:
        magnitude of point
    """
    return math.sqrt(point[0]**2 + point[1]**2)


def dist_to_line(start: Point, end: Point, *points: Point, signed=False) -> Union[float, List[float]]:
    """Finds the distance between points and a line given by two points

    Args:
        start: first point for line
        end: second point for line
        points: points to find distance of. 

    Returns:
        A single distance if only one point is provided, otherwise a list of distances.
    """
    start_x, start_y = start 
    end_x, end_y = end 

    dy = end_y - start_y 
    dx = end_x - start_x 
    m_dif = end_x*start_y - end_y*start_x
    denom = math.sqrt(dy**2 + dx**2)

    _dist = lambda point: (dy*point[0] - dx*point[1] + m_dif) / denom
    dist = _dist if signed else lambda point: abs(_dist(point))

    if len(points) == 1:
        return dist(points[0])
    else:
        return list(map(dist, points))