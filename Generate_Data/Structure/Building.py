import numpy as np
import shapely
from shapely import geometry, affinity


def read_vertices(poly_points):

    x_list = poly_points[0][:-1]
    y_list = poly_points[1][:-1]

    return np.array((x_list, y_list))


class Building:

    def __init__(self, cx, cy, w, h, angle, id):
        self.cx = cx
        self.cy = cy
        self.w = w
        self.h = h
        self.angle = angle
        self.id = id
        self.center = np.array((cx, cy))

        c = shapely.geometry.box(-w / 2.0, -h / 2.0, w / 2.0, h / 2.0)
        rc = shapely.affinity.rotate(c, self.angle, origin='centroid')
        rce = shapely.affinity.translate(rc, self.cx, self.cy)
        exterior_points = rce.exterior.coords.xy
        self.vertices = read_vertices(exterior_points)

    def offset_center(self, vector):
        self.center += vector

    def get_contour(self):
        w = self.w
        h = self.h
        c = shapely.geometry.box(-w / 2.0 - 2, -h / 2.0 - 2, w / 2.0 + 2, h / 2.0 + 2)
        rc = shapely.affinity.rotate(c, self.angle, origin='centroid')
        return shapely.affinity.translate(rc, self.cx, self.cy)

    def intersection(self, other):
        return self.get_contour().intersection(other.get_contour())
