import cvxpy as cp
import numpy as np
import matplotlib.pyplot as plt
import glob
import math
import mosek
from lanes_generator import plot_shape, read_shape
from more_itertools import distinct_combinations

de = 1.0  # Expected spacing between cones #TODO Should be changed based on the rules of the competition
dt = 0.75  # Tunable threshold parameter #TODO Should be changed based on the real expected spacing
theta_e = math.pi  # Expected angle between two adjacent edges; setting it to pi proved to be effective
theta_t = 3 / 4 * math.pi  # TODO Maybe it should be changed too
ws = 2  # Spacing weight #TODO This should be changed too
wt = 2  # Angle cost weight #TODO this should be changed too
wb = 10  # Uniform benefit of adding an edge
s_crit = 1  # Maximum allowed spacing cost   #TODO This should be changed too
t_crit = 1  # Maximum allowed angle cost   #TODO This should be changed too
dmin = 3  # Minimum width

NUM_CONES = 10  # Bubuie daca e mai mare de 10 si nu stiu de ce

## Returns the coordinates of the center of an edge
def line_center(ci, cj):
    return zip((ci[0] + cj[0]) * 0.5, (ci[1] + cj[1]) * 0.5)


## Returns 1 if the minimum euclidean distance connects the endpoints of 2 segments, 0 otherwise
## See https://imgur.com/a/FGTWzrC for a visual interpretation
def end_dist(s1, s2):
    if s1 is None or s2 is None:
        return None

    values = []

    # there are 8 cases to be taken into consideration:
    # 4 for endpoints + 4 for invalid minimums
    # 1 - 4
    for i in s1:
        for j in s2:
            values.append(euclidean_distance(i, j))

    current_min = min(values)

    # 5 - 8
    center1 = line_center(s1[0], s1[1])
    center2 = line_center(s2[0], s2[1])
    for i in s1:
        if euclidean_distance(center1, i) < current_min:
            return 0
    for j in s2:
        if euclidean_distance(center2, j) < current_min:
            return 0

    return 1


# Extract the cones as an array of pairs (x, y) from the given shape.
# Both inner cones and outer cones are in the same array, hence the algorithm
# does not differentiate between them.
# We take one inner lane cone and one outer lane cone to ensure that cones are taken from both paths. This is not
# required if we would have been used all the lanes, but is very time-consuming so we use only NUM_CONES.
def get_cones(shape):
    inner_lanes = shape['inner_lanes']
    outer_lanes = shape['outer_lanes']
    cones = []
    for i in range(len(inner_lanes)):
        cones.append(inner_lanes[i])
        cones.append(outer_lanes[i])
    return cones


# Returns the vectorized representation of the lower triangular portion of matrix M in column-major order
def vectorize_matrix(M):
    n = len(M)

    v = []
    for j in range(n):
        for i in range(j + 1, n):
            v.append(M[i][j])
    return np.array(v)


# Computes the index of the vectorized representation of a n x n matrix given row i and column j
# If the index is invalid returns None - should be checked for None value
def get_idx(i, j, n):
    if j > i:
        aux = i
        i = j
        j = aux

    if j == i:
        return None

    column_offset = 0
    for k in range(1, j + 1):
        column_offset += n - k

    if column_offset >= 0.5 * (n - 1) * n:
        return None

    if not (j + 1 <= i <= n - 1):
        return None

    row_offset = i - (j + 1)

    return column_offset + row_offset


# Computes the index in the vector f
def get_idx_three_params(i, j, k, n):
    if j > i:
        aux = i
        i = j
        j = aux

    if j == i:
        return None

    column_offset = 0
    for temp in range(1, j + 1):
        column_offset += (n - temp) * (n - 2)

    if column_offset >= 0.5 * (n - 2) * (n - 1) * n:
        return None

    if not (j + 1 <= i <= n - 1):
        return None

    row_offset = (i - (j + 1)) * (n - 2)

    k_offset = k
    if k > i:
        k_offset -= 1

    if k > j:
        k_offset -= 1

    return column_offset + row_offset + k_offset


#  Computes the euclidean distance between to cones
#  Input: two cones, each cone represented as a pair (x, y) denoting its position
def euclidean_distance(ci, cj):
    return math.sqrt((ci[0] - cj[0]) ** 2 + (ci[1] - cj[1]) ** 2)


# Returns the angle given by 3 points with c1 being the vertex corresponding to the angle
# All the points are represented as a pair (x, y) denoting their position
# Formula taken from https://stackoverflow.com/questions/1211212/how-to-calculate-an-angle-from-three-points
def angle(c1, c2, c3):
    d12 = euclidean_distance(c1, c2)
    d13 = euclidean_distance(c1, c3)
    d23 = euclidean_distance(c2, c3)
    return math.acos((d12 ** 2 + d13 ** 2 - d23 ** 2) / (2 * d12 * d13))


# Could have been optimized but I'm lazy
def compute_distance_matrix(cones):
    n = len(cones)
    D = np.zeros((n, n))
    for i, j in zip(range(n), range(n)):
        D[i][j] = euclidean_distance(cones[i], cones[j])
    return D


def compute_spacing_cost(D):
    s = np.zeros(D.shape)
    for i in range(D.shape[0]):
        for j in range(D.shape[1]):
            s[i][j] = ((D[i][j] - de) / dt) ** 4

    return s


def compute_angle_cost(cones):
    n = len(cones)

    t = []
    for j in range(n):
        for i in range(j + 1, n):
            for k in range(n):
                if k != i and k != j:
                    theta = angle(cones[j], cones[i], cones[k])
                    t.append(((theta - theta_e) / theta_t) ** 4)
    return np.array(t)


# Returns a list of all subsets of a given set
def powerset(s):
    x = len(s)
    p = []
    for i in range(1 << x):
        p += [[s[j] for j in range(x) if (i & (1 << j))]]

    return p


def lane_detection(cones):
    n = len(cones)
    na = int(0.5 * (n - 1) * n)  # Number of elements of a
    nf = int(0.5 * (n - 2) * (n - 1) * n)  # Number of elements of f

    D = compute_distance_matrix(cones)

    s = vectorize_matrix(compute_spacing_cost(D))  # s is the vectorized spacing cost
    t = compute_angle_cost(cones)  # t is the vectorized angle cost

    sc = [1 if s[i] > s_crit else 0 for i in range(len(s))]
    tc = [1 if t[i] > t_crit else 0 for i in range(len(t))]

    j = np.ones(na)

    s.reshape(na, 1)
    j.reshape(na, 1)
    t.reshape(nf, 1)

    a = cp.Variable((na,), boolean=True)
    f = cp.Variable((nf,), boolean=True)
    g = cp.Variable((n,), boolean=True)

    # The objective is to minimize the cost function
    objective = cp.Minimize((ws * s - wb * j).T @ a + wt * t.T @ f)

    # Cost Constraints
    constraints = [cp.multiply(sc, a) <= np.ones(na), cp.multiply(tc, f) <= np.ones(nf)]

    for j in range(n):
        for i in range(j + 1, n):
            for k in range(n):
                if k != i and k != j:
                    ji = get_idx(j, i, n)
                    jk = get_idx(j, k, n)
                    ijk = get_idx_three_params(i, j, k, n)

                    constraints.append(2 * f[ijk] - a[ji] - a[jk] <= 0)
                    constraints.append(f[ijk] - a[ji] - a[jk] >= -1)

    A = []
    for i in range(n):
        Ai = cp.sum([a[get_idx(j, i, n)] if i != j else 0 for j in range(n)])
        A.append(Ai)
        constraints.append(g[i] - Ai <= 0)
        constraints.append(1 / 2 * Ai - g[i] <= 0)

    constraints.append(cp.sum([A[i] - 2 * g[i] for i in range(n)]) == -4)

    # To ensure there are at least 2 disjoint subgraphs
    constraints.append(0.5 * cp.sum([A[i] for i in range(n)]) <= cp.sum([g[i] for i in range(n)]) - 2)

    # to guarantee a lane graph
    cone_set = [i for i in range(n)]
    subsets = powerset(cone_set)

    # something fishy here - changed
    for s in subsets:
        for i in s:
            set_sum = cp.sum([a[get_idx(i, j, n)] if i != j else 0 for j in s])
            constraints.append(set_sum <= len(s) - 1)

    # pairwise edge constraints
    for i, j in zip(range(n), range(n)):
        for k, l in zip(range(n), range(n)):
            if k != (i or j) and l != (i or j) and k != l:
                ij = get_idx(i, j, n)
                kl = get_idx(k, l, n)
                constraints.append(a[ij] + a[kl] <= 1)
                constraints.append(euclidean_distance(zip(i, j), zip(k, l)) > dmin and end_dist(zip(i, j), zip(k, l)))

    problem = cp.Problem(objective, constraints)
    problem.solve(solver=cp.MOSEK)

    print("status:", problem.status)
    print("optimal value", problem.value)
    print("optimal var", a.value)

    return np.copy(a.value)


def plot_boundary(a, shape):
    print('a', a)
    fig, ax = plt.subplots()
    ax.set_aspect("equal")

    cones = get_cones(shape)[:NUM_CONES]
    n = len(cones)

    for j in range(n):
        for i in range(j + 1, n):
            idx = get_idx(i, j, n)
            if a[idx] == 1:
                xi, yi = cones[i]
                xj, yj = cones[j]

                plt.plot([xi, xj], [yi, yj], 'k-')

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


def main():
    shapes = glob.glob('../shapes/shape_*')
    shapes_counter = len(shapes)

    ans = input("Which shape would you like to load?: ")
    shape_number = int(ans)

    if 0 <= shape_number <= shapes_counter:
        print('Loading shape ' + ans)
        shape = read_shape('../shapes/shape_' + ans)
    else:
        shape = read_shape('../shapes/shape_0')

    # plot_shape(shape)

    # Cannot insert all the cones because it runs very slow
    cones = get_cones(shape)[0:NUM_CONES]
    # print(cones)

    a = lane_detection(cones)  # What should be called in the end to obtain the result

    plot_boundary(a, shape)


if __name__ == "__main__":
    main()
