import numpy as np
import matplotlib.pyplot as plt
from lanes_generator import plot_shape
from boundary_detection import get_cones, lane_detection, get_idx, plot_boundary

START = 1
STOP = 10
NUM_POINTS = 25
NUM_CONES = 10
STEP = 3  # To create distance distance between cone pairs
OVERLAP_SPACING = 2  # To avoid overlapping points near the corners


# Returns an array of coordinates [[x1, y1], [x2, y2] ...]
def points_generator():
    base_coords = [round(i, 5) for i in np.linspace(START, STOP, NUM_POINTS)]
    points = []

    # 1
    tmp_egde = [[START, i] for i in base_coords]
    points += tmp_egde

    # 2
    tmp_egde = [[i, STOP] for i in base_coords]
    points += tmp_egde

    # 3
    tmp_egde = [[STOP, i] for i in base_coords]
    tmp_egde.reverse()
    points += tmp_egde

    # 4
    tmp_egde = [[i, START] for i in base_coords]
    tmp_egde.reverse()
    points += tmp_egde

    return points


def resize_lane(coordinates, width):
    lane = []

    for i in range(START, len(coordinates), STEP):
        tmp = coordinates[i]
        # first edge
        if i > START + OVERLAP_SPACING and i <= 25 - OVERLAP_SPACING:
            lane.append((tmp[0] + width, tmp[1]))
        # second edge
        elif i > 25 and i < 50 - OVERLAP_SPACING:
            lane.append((tmp[0], tmp[1] - width))
        # third edge
        elif i > 50 and i < 75 - OVERLAP_SPACING:
            lane.append((tmp[0] - width, tmp[1]))
        # fourth edge
        elif i > 75 and i < 99:
            lane.append((tmp[0], tmp[1] + width))
    return lane


def lanes_generator(coordinates, width):
    inner_lanes = resize_lane(coordinates, width)
    outer_lanes = resize_lane(coordinates, -width)

    return inner_lanes, outer_lanes


def square_generator():
    shape = {}

    base_coords = points_generator()

    shape['x_curve'] = [x for [x, y] in base_coords]
    shape['y_curve'] = [y for [x, y] in base_coords]

    inner_lanes, outer_lanes = lanes_generator(base_coords, 0.5)

    shape['inner_lanes'] = inner_lanes
    shape['outer_lanes'] = outer_lanes

    return shape


def main():
    shape = square_generator()
    cones = get_cones(shape)[17:NUM_CONES + 17]
    a = lane_detection(cones)
    plot_boundary(a, shape, cones)


if __name__ == "__main__":
    main()
