import math

DEGREES_TO_RADIANS = math.pi / 180
RADIANS_TO_DEGREES = 1 / DEGREES_TO_RADIANS
EARTH_MEAN_RADIUS_METER = 6378137
DEG_TO_KM = DEGREES_TO_RADIANS * EARTH_MEAN_RADIUS_METER
LAT_PER_METER = 8.993203677616966e-06
LNG_PER_METER = 1.1700193970443768e-05


class SPoint:
    def __init__(self, lat, lng):
        self.lat = lat
        self.lng = lng

    def __str__(self):
        return '({},{})'.format(self.lat, self.lng)

    def __repr__(self):
        return self.__str__()

    def __eq__(self, other):
        # equal. Orginally is compared with reference. Here we change to value
        return self.lat == other.lat and self.lng == other.lng

    def __ne__(self, other):
        # not equal
        return not self == other

    def __hash__(self):
        return hash(str(self.lat) + " " + str(self.lng))


def same_coords(a, b):
    # we can directly use == since SPoint has updated __eq__()
    if a == b:
        return True
    else:
        return False


def distance(a, b):
    """
    Calculate haversine distance between two GPS points in meters
    Args:
    -----
        a,b: SPoint class
    Returns:
    --------
        d: float. haversine distance in meter
    """
    if same_coords(a, b):
        return 0.0
    delta_lat = math.radians(b.lat - a.lat)
    delta_lng = math.radians(b.lng - a.lng)
    h = math.sin(delta_lat / 2.0) * math.sin(delta_lat / 2.0) + math.cos(math.radians(a.lat)) * math.cos(
        math.radians(b.lat)) * math.sin(delta_lng / 2.0) * math.sin(delta_lng / 2.0)
    c = 2.0 * math.atan2(math.sqrt(h), math.sqrt(1 - h))
    d = EARTH_MEAN_RADIUS_METER * c
    return d


# http://www.movable-type.co.uk/scripts/latlong.html
def bearing(a, b):
    """
    Calculate the bearing of ab
    """
    pt_a_lat_rad = math.radians(a.lat)
    pt_a_lng_rad = math.radians(a.lng)
    pt_b_lat_rad = math.radians(b.lat)
    pt_b_lng_rad = math.radians(b.lng)
    y = math.sin(pt_b_lng_rad - pt_a_lng_rad) * math.cos(pt_b_lat_rad)
    x = math.cos(pt_a_lat_rad) * math.sin(pt_b_lat_rad) - math.sin(pt_a_lat_rad) * math.cos(pt_b_lat_rad) * math.cos(
        pt_b_lng_rad - pt_a_lng_rad)
    bearing_rad = math.atan2(y, x)
    return math.fmod(math.degrees(bearing_rad) + 360.0, 360.0)


def cal_loc_along_line(a, b, rate):
    """
    convert rate to gps location
    """
    lat = a.lat + rate * (b.lat - a.lat)
    lng = a.lng + rate * (b.lng - a.lng)
    return SPoint(lat, lng)


def project_pt_to_segment(a, b, t):
    """
    Args:
    -----
    a,b: start/end GPS location of a road segment
    t: raw point
    Returns:
    -------
    project: projected GPS point on road segment
    rate: rate of projected point location to road segment
    dist: haversine_distance of raw and projected point
    """
    ab_angle = bearing(a, b)
    at_angle = bearing(a, t)
    ab_length = distance(a, b)
    at_length = distance(a, t)
    delta_angle = at_angle - ab_angle
    meters_along = at_length * math.cos(math.radians(delta_angle))
    if ab_length == 0.0:
        rate = 0.0
    else:
        rate = meters_along / ab_length
    if rate >= 1:
        projection = SPoint(b.lat, b.lng)
        rate = 1.0
    elif rate <= 0:
        projection = SPoint(a.lat, a.lng)
        rate = 0.0
    else:
        projection = cal_loc_along_line(a, b, rate)
    dist = distance(t, projection)
    return projection, rate, dist


import numpy as np


def project_pt_to_road(rn, t, rid):
    """
    Args:
    -----
    rn: road_network
    t: raw point
    rid: road edge id
    Returns:
    -------
    project: projected GPS point on road segment
    rate: rate of projected point location to road segment
    dist: haversine_distance of raw and projected point
    """
    edge_cords = rn.edgeCord[rid]
    dis = [distance(t, SPoint(edge_cords[2 * i], edge_cords[2 * i + 1])) for i in range(len(edge_cords) // 2)]
    idx = np.argmin(dis)
    candidate = []
    if idx != 0:
        candidate.append([*project_pt_to_segment(SPoint(edge_cords[2 * (idx - 1)], edge_cords[2 * (idx - 1) + 1]),
                                                 SPoint(edge_cords[2 * idx], edge_cords[2 * idx + 1]), t), idx])
    if idx != len(edge_cords) // 2 - 1:
        candidate.append([*project_pt_to_segment(SPoint(edge_cords[2 * idx], edge_cords[2 * idx + 1]),
                                                 SPoint(edge_cords[2 * (idx + 1)], edge_cords[2 * (idx + 1) + 1]), t),
                          idx + 1])
    best_candidate = candidate[0]
    if len(candidate) == 2 and candidate[0][2] > candidate[1][2]:
        best_candidate = candidate[1]
    projection, rate, dist, idx = best_candidate
    dist_to_end = (1 - rate) * distance(SPoint(edge_cords[2 * (idx - 1)], edge_cords[2 * (idx - 1) + 1]),
                                        SPoint(edge_cords[2 * idx], edge_cords[2 * idx + 1])) + rn.edgeOffset[rid][idx]
    if rn.edgeDis[rid] > 0:
        return projection, 1 - (dist_to_end / rn.edgeDis[rid]), dist
    else:
        return projection, 1, dist


def rate2gps(rn, rid, rate) -> SPoint:
    """
    Convert road rate to GPS on the road segment.
    Since one road contains several coordinates, iteratively computing length can be more accurate.
    Args:
    -----
    rn: road network
    rid, rate: single value from model prediction
    Returns:
    --------
    project_pt:
        projected GPS point on the road segment.
    """

    cords = np.array(rn.edgeCord[rid]).reshape(-1, 2).tolist()
    offset = rn.edgeDis[rid] * rate
    dist = 0  # temp distance for coords
    pre_dist = 0  # coords distance is smaller than offset

    if rate == 1.0:
        return SPoint(*cords[-1])
    if rate == 0.0:
        return SPoint(*cords[0])

    project_pt = SPoint(*cords[0])

    for i in range(len(cords) - 1):
        if i > 0:
            pre_dist += distance(SPoint(*cords[i - 1]), SPoint(*cords[i]))
        dist += distance(SPoint(*cords[i]), SPoint(*cords[i + 1]))
        if dist >= offset:
            if distance(SPoint(*cords[i]), SPoint(*cords[i + 1])) < 1e-6:  # zero segment length
                coor_rate = 0
            else:
                coor_rate = (offset - pre_dist) / distance(SPoint(*cords[i]), SPoint(*cords[i + 1]))
            project_pt = cal_loc_along_line(SPoint(*cords[i]), SPoint(*cords[i + 1]), coor_rate)
            break

    return project_pt
