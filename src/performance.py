import numpy as np
from boundary_detection import get_cones, lane_detection, get_idx, plot_boundary
import pickle, glob

# USED FOR THE RECTANGLE CASE
START = 1
STOP = 10
RECT_POINTS = 25  # Used to create the base coordinates
NUM_CONES = 10  # Shady business above 10
STEP = 3  # To create distance distance between cone pairs
OVERLAP_SPACING = 2  # To avoid overlapping points near the corners

# USED FOR THE CIRCLE CASE
CIRCLE_POINTS = 50  # Used to create the base coordinates; more points -> a more accurate circle
BASE_RADIUS = 5  # Used to create the base circle
SPACING = 2  # Used to create the inner and outer lanes
CONE_SHIFT = 10  # Used to shift position among the cones


# Creates a simple circle, given a radius
def base_circle(radius):
    # theta goes from 0 to 2pi
    theta = np.linspace(0, 2 * np.pi, CIRCLE_POINTS)

    # the radius of the circle
    r = np.sqrt(radius)

    # compute x1 and x2
    x = r * np.cos(theta)
    y = r * np.sin(theta)

    return x, y


# Creates a neighbouring circular lane, given a certain distance (width)
# Returns the lane as an array of cones
def circle_compute_lane(radius, width):
    x, y = base_circle(radius + width)

    lane = []
    for i in range(1, len(x), STEP):
        lane.append((x[i], y[i]))

    return lane


# Returns an array of coordinates [[x1, y1], [x2, y2] ...] used for a rectangle
def rect_points_generator():
    base_coords = [round(i, 5) for i in np.linspace(START, STOP, RECT_POINTS)]
    points = []

    # 1
    tmp_edge = [[START, i] for i in base_coords]
    points += tmp_edge

    # 2
    tmp_edge = [[i, STOP] for i in base_coords]
    points += tmp_edge

    # 3
    tmp_edge = [[STOP, i] for i in base_coords]
    tmp_edge.reverse()
    points += tmp_edge

    # 4
    tmp_edge = [[i, START] for i in base_coords]
    tmp_edge.reverse()
    points += tmp_edge

    return points


# Creates a neighbouring lane given a base lane (coordinates) and a distance (width)
def rect_compute_lane(coordinates, width):
    lane = []

    for i in range(0, len(coordinates), STEP):
        tmp = coordinates[i]
        # first edge
        if OVERLAP_SPACING < i <= 25 - OVERLAP_SPACING:
            lane.append((tmp[0] + width, tmp[1]))
        # second edge
        elif 25 < i < 50 - OVERLAP_SPACING:
            lane.append((tmp[0], tmp[1] - width))
        # third edge
        elif 50 < i < 75 - OVERLAP_SPACING:
            lane.append((tmp[0] - width, tmp[1]))
        # fourth edge
        elif 75 < i < 99:
            lane.append((tmp[0], tmp[1] + width))
    return lane


def square_generator():
    shape = {}

    base_coords = rect_points_generator()
    shape['x_curve'] = [x for [x, y] in base_coords]
    shape['y_curve'] = [y for [x, y] in base_coords]

    # creating the inner and outer lanes
    width = 0.5
    inner_lanes = rect_compute_lane(base_coords, width)
    outer_lanes = rect_compute_lane(base_coords, -width)
    shape['inner_lanes'] = inner_lanes
    shape['outer_lanes'] = outer_lanes

    return shape


def circle_generator():
    shape = {}

    shape['x_curve'], shape['y_curve'] = base_circle(BASE_RADIUS)
    shape['inner_lanes'] = circle_compute_lane(BASE_RADIUS, -SPACING)
    shape['outer_lanes'] = circle_compute_lane(BASE_RADIUS, SPACING)

    return shape


def rectangle_performance():
    shape = square_generator()
    cones = get_cones(shape)[17:NUM_CONES + 17]
    a = lane_detection(cones)
    plot_boundary(a, shape, cones)


def circle_performance():
    shape = circle_generator()
    cones = get_cones(shape)[CONE_SHIFT:NUM_CONES + CONE_SHIFT]
    a = lane_detection(cones)
    plot_boundary(a, shape, cones)


def main():
    ans = input("Which shape would you like to test? Please choose between circle and rectangle:")
    if ans.lower() == 'circle':
        circle_performance()
    elif ans.lower() == 'rectangle':
        rectangle_performance()

    print("Exit")


if __name__ == "__main__":
    main()
