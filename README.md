# epsm-lane-optimization

The repository contains a custom implementation of [this](https://static1.squarespace.com/static/5b79970e3c3a53723fab8cfc/t/5dd31c148feff15f97f4ddbc/1574116375972/ICRA2020_Boundary_Gen_final.pdf) Binary Integer Optimization Algorithm.

## Structure 
The project is structured into three main modules: 
1. _**lanes_generator.py**_: the module is used to create random Formula One tracks with cones on each side
2. _**performance.py**_: used to check the behaviour of the algorithm on given tests
3. _**boundary_detection.py**_: receives a given track and performs The
Robust Lane Detection optimization (RLD) on it

## Usage 
In _lanes_generator.py_, a random track will be generated. If you want to save the shape,
answer `yes` in the terminal. The shape will automatically be saved in the
_shapes_ directory.

In _boundary_detection.py_, write in the terminal the index
of the shape you would like to load. Simply write a numeric value (such as `0`, for shape 0) to load
the desired shape. The algorithm will compute the boundary and show it in a plot. 

## The Robust Lane Detection optimization
The algorithm provides a solution to real-time lane detection, in the context of the Formula Student
Driverless racecar competition. \
We represent the boundaries of the lane with an undirected graph _G(V, E)_ where the vertices V are the set
of observed cones, including false positives, and E is the set of edges connecting cones that are adjacent
to each other. The overall strategy is to solve a **constrained binary integer program** to and the adjacency
matrix, A, of the lane graph that is optimal with respect to a specified cost function. \
The binary integer optimization problem was modelled using [cvxpy](https://www.cvxpy.org/), a Python-embedded modeling language for convex optimization problems.
The solver used was [Mosek](https://www.mosek.com/).

## Results 
An example of the result obtained for a random track is given in the following image:
![lanes_0](https://user-images.githubusercontent.com/48925470/117624636-ca6b7380-b17d-11eb-8d7c-60c8f265e4ea.png)

