import math
import numpy as np
from scipy.special import binom
import matplotlib.pyplot as plt
import pickle
import glob

bernstein = lambda n, k, t: binom(n, k) * t ** k * (1. - t) ** (n - k)


def bezier(points, num=200):
    N = len(points)
    t = np.linspace(0, 1, num=num)
    curve = np.zeros((num, 2))
    for i in range(N):
        curve += np.outer(bernstein(N - 1, i, t), points[i])
    return curve


class Segment:
    def __init__(self, p1, p2, angle1, angle2, **kw):
        self.p1 = p1;
        self.p2 = p2
        self.angle1 = angle1;
        self.angle2 = angle2
        self.numpoints = kw.get("numpoints", 100)
        r = kw.get("r", 0.3)
        d = np.sqrt(np.sum((self.p2 - self.p1) ** 2))
        self.r = r * d
        self.p = np.zeros((4, 2))
        self.p[0, :] = self.p1[:]
        self.p[3, :] = self.p2[:]
        self.calc_intermediate_points(self.r)

    def calc_intermediate_points(self, r):
        self.p[1, :] = self.p1 + np.array([self.r * np.cos(self.angle1),
                                           self.r * np.sin(self.angle1)])
        self.p[2, :] = self.p2 + np.array([self.r * np.cos(self.angle2 + np.pi),
                                           self.r * np.sin(self.angle2 + np.pi)])
        self.curve = bezier(self.p, self.numpoints)


def get_curve(points, **kw):
    segments = []
    for i in range(len(points) - 1):
        seg = Segment(points[i, :2], points[i + 1, :2], points[i, 2], points[i + 1, 2], **kw)
        segments.append(seg)
    curve = np.concatenate([s.curve for s in segments])
    return segments, curve


def ccw_sort(p):
    d = p - np.mean(p, axis=0)
    s = np.arctan2(d[:, 0], d[:, 1])
    return p[np.argsort(s), :]


def get_bezier_curve(a, rad=0.2, edgy=0):
    """ given an array of points *a*, create a curve through
    those points.
    *rad* is a number between 0 and 1 to steer the distance of
          control points.
    *edgy* is a parameter which controls how "edgy" the curve is,
           edgy=0 is smoothest."""
    p = np.arctan(edgy) / np.pi + .5
    a = ccw_sort(a)
    a = np.append(a, np.atleast_2d(a[0, :]), axis=0)
    d = np.diff(a, axis=0)
    ang = np.arctan2(d[:, 1], d[:, 0])
    f = lambda ang: (ang >= 0) * ang + (ang < 0) * (ang + 2 * np.pi)
    ang = f(ang)
    ang1 = ang
    ang2 = np.roll(ang, 1)
    ang = p * ang1 + (1 - p) * ang2 + (np.abs(ang2 - ang1) > np.pi) * np.pi
    ang = np.append(ang, [ang[0]])
    a = np.append(a, np.atleast_2d(ang).T, axis=1)
    s, c = get_curve(a, r=rad, method="var")
    x, y = c.T
    return x, y, a


def get_random_points(n=5, scale=0.8, mindst=None, rec=0):
    """ create n random points in the unit square, which are *mindst*
    apart, then scale them."""
    mindst = mindst or .7 / n
    a = np.random.rand(n, 2)
    d = np.sqrt(np.sum(np.diff(ccw_sort(a), axis=0), axis=1) ** 2)
    if np.all(d >= mindst) or rec >= 200:
        return a * scale
    else:
        return get_random_points(n=n, scale=scale, mindst=mindst, rec=rec + 1)


def lanes_generator(x, y, dist=0.5):
    outer_lanes = []
    inner_lanes = []

    neigh = 15

    for i in range(0, len(x) - neigh - 1, neigh):
        A = (x[i], y[i])
        B = (x[i + neigh], y[i + neigh])

        C = ((A[0] + B[0]) / 2, (A[1] + B[1]) / 2)

        if abs(B[1] - A[1]) < 0.01:
            x_o = x_i = C[0]
            y_i = C[1] - dist
            y_o = C[1] + dist
        elif abs(B[0] - A[0]) < 0.01:
            y_o = y_i = C[1]
            x_i = C[0] + dist
            x_o = C[0] - dist
        else:
            d1_m = (B[1] - A[1]) / (B[0] - A[0])
            d2_m = -  1 / d1_m
            d2_b = C[1] - d2_m * C[0]

            x_c, y_c = C

            # Solve the quadratic equation
            a = (1 + d2_m ** 2)
            b = 2 * (d2_m * (d2_b - y_c) - x_c)
            c = x_c ** 2 + (d2_b - y_c) ** 2 - dist ** 2

            x_i = (-b + math.sqrt(b ** 2 - 4 * a * c)) / (2 * a)
            y_i = d2_m * x_i + d2_b

            x_o = (-b - math.sqrt(b ** 2 - 4 * a * c)) / (2 * a)
            y_o = d2_m * x_o + d2_b

        outer_lanes.append((x_o, y_o))
        inner_lanes.append((x_i, y_i))

    return inner_lanes, outer_lanes


def plot_shape(shape):
    fig, ax = plt.subplots()
    ax.set_aspect("equal")

    x = shape['x_curve']
    y = shape['y_curve']

    plt.plot(x, y)

    inner_lanes = shape['inner_lanes']
    outer_lanes = shape['outer_lanes']

    x = [xi for xi, _ in outer_lanes]
    y = [yi for _, yi in outer_lanes]
    plt.plot(x, y, 'go')

    x = [xi for xi, _ in inner_lanes]
    y = [yi for _, yi in inner_lanes]
    plt.plot(x, y, 'go')

    plt.show()


def generate_shape():
    shape = {}

    rad = 0.5
    edgy = 0.05

    c = [0, 0]

    random_points = get_random_points(n=4, scale=10)

    a = random_points + c
    x, y, _ = get_bezier_curve(a, rad=rad, edgy=edgy)

    shape['x_curve'] = x
    shape['y_curve'] = y

    inner_lanes, outer_lanes = lanes_generator(x, y, 0.5)

    shape['inner_lanes'] = inner_lanes
    shape['outer_lanes'] = outer_lanes

    return shape


# The path to the shape dir depends on your project structure
# The current structure is:
# |--shapes/
# |--src/\
#     |--lanes_generator.py
def save_shape(shape):
    shapes = glob.glob('../shapes/shape_*')
    shapes_counter = len(shapes)

    filename = '../shapes/shape_' + str(shapes_counter)
    outfile = open(filename, 'wb')

    pickle.dump(shape, outfile)
    outfile.close()


def read_shape(filename):
    infile = open(filename, 'rb')
    shape = pickle.load(infile)
    infile.close()

    return shape


def main():
    shape = generate_shape()
    plot_shape(shape)

    ans = input("Would you like to save this shape?: ")
    if ans.lower() == 'yes':
        print('Saving shape...')
        save_shape(shape)

    print("Exit")


if __name__ == "__main__":
    main()
