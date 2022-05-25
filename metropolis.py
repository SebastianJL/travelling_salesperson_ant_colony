# Travelling salesman problem -> find shortest route

# Imports

import numpy as np
from numba import jit
from matplotlib import pyplot as plt
from typing import List, Tuple
import time


# Functions


# Reading the tsp problem
def read_problem_input(file_name: str) -> List[Tuple[float, float]]:
    cities = []
    with open(file_name, "r") as f:
        for line in f:
            line = line.strip()
            if line == "":
                continue
            if line[0].isdigit():
                try:
                    t, x, y, *_ = line.split()
                except ValueError:
                    x, y, *_ = line.split()
                cities.append((float(x.strip()), float(y.strip())))
    return cities


# Plotting the path
def plot_path(cities: List[Tuple[float, float]], path: List[int], ax=None):
    if ax is None:
        ax = plt.gca()
    xs = [c[0] for c in cities]
    ys = [c[1] for c in cities]
    ax.scatter(xs, ys, c="k")
    for i in range(len(path)):
        x1, y1 = cities[path[i]]
        x2, y2 = cities[path[(i + 1)%len(path)]]
        ax.plot([x1, x2], [y1, y2], c="r")
    ax.axis("equal")
    plt.show()


# Distance between 2 cities
def dist(a, b):
    return np.sqrt((b[0] - a[0])**2 + (b[1] - a[1])**2)


# Total distance of a path
def tot_energy(tour, cities):
    E = 0
    for i, t in enumerate(tour):
        E += dist(cities[tour[i]], cities[tour[i-1]])

    return E


# Randomly swap 2 cities in a tour (this doesn't work!)
def rand_swap(tour, cities, beta):
    a = np.random.choice(tour)
    i = np.where(tour == a)[0][0]
    b = np.random.choice(tour)
    j = np.where(tour == b)[0][0]

    if i == j:
        print("stopped")
        return 0, tour

    if i+1 == len(tour):
        i = -1
    if j+1 == len(tour):
        j = -1

    i_left = i-1
    i_right = i+1
    j_left = j-1
    j_right = j+1

    i_left_new = j-1
    i_right_new = j+1
    j_left_new = i-1
    j_right_new = i+1

    if j-1 == i or (j == -1 & i == len(tour)-2) or (i == -1 & j == 0):
        i_left_new = j  #j
        j_right_new = i #i
    if i-1 == j or (i == -1 & j == len(tour)-2) or (j == -1 & i == 0):
        j_left_new = i  #j-2
        i_right_new = j #i+2

    if tour[i] == tour[i_right_new] or tour[i] == tour[i_left_new]:
        print("weird stuff happening i")
        print(f"i {i}, r {i_right_new}, l {i_left_new}")
        print(f"c {tour[i]}, r{tour[i_right_new]}, l {tour[i_left_new]}")
    if tour[j] == tour[j_right_new] or tour[j] == tour[j_left_new]:
        print("weird stuff happening j")
        print(f"j {j}, r {j_right_new}, l {j_left_new}")
        print(f"c {tour[j]}, r{tour[j_right_new]}, l {tour[j_left_new]}")


    e_old = dist(cities[tour[i_left]], cities[tour[i]]) + \
            dist(cities[tour[i]], cities[tour[i_right]]) + \
            dist(cities[tour[j_left]], cities[tour[j]]) + \
            dist(cities[tour[j]], cities[tour[j_right]])

    e_new = dist(cities[tour[j_left_new]], cities[tour[j]]) + \
            dist(cities[tour[j]], cities[tour[j_right_new]]) + \
            dist(cities[tour[i_left_new]], cities[tour[i]]) + \
            dist(cities[tour[i]], cities[tour[i_right_new]])

    print(dist(cities[tour[j_left_new]], cities[tour[j]]))
    print(dist(cities[tour[j]], cities[tour[j_right_new]]))
    print(dist(cities[tour[i_left_new]], cities[tour[i]]))
    print(dist(cities[tour[i]], cities[tour[i_right_new]]))

    # TODO: E values get below 0, check where the error lies (dist, swap...)

    dE = e_new - e_old
    print(f"dE: {dE}")

    if dE < 0:
        tour[i] = b
        tour[j] = a
    else:
        r = np.random.random()
        if r < np.exp(-beta * dE):
            tour[i] = b
            tour[j] = a
        else:
            dE = 0

    return dE, tour


# Randomly reverse a section in a tour
def rand_reverse(old_tour, cities, beta):
    a = np.random.choice(old_tour)
    b = np.random.choice(old_tour)
    i = min(np.where(old_tour == a)[0][0], np.where(old_tour == b)[0][0])
    j = max(np.where(old_tour == a)[0][0], np.where(old_tour == b)[0][0])

    if i == j:
        return 0, old_tour

    if j+1 == len(old_tour):
        j = -1

    seg = old_tour[i:j+1]
    if len(seg) == len(old_tour) or len(seg) == 0 or len(seg) == 1:
        return 0, old_tour

    e_old = dist(cities[old_tour[i-1]], cities[old_tour[i]]) \
            + dist(cities[old_tour[j]], cities[old_tour[j+1]])

    e_new = dist(cities[old_tour[i-1]], cities[old_tour[j]]) \
            + dist(cities[old_tour[i]], cities[old_tour[j+1]])

    dE = e_new - e_old

    new_tour = old_tour

    if dE < 0:
        new_tour[i:j+1] = seg[::-1]
    else:
        r = np.random.random()
        if r < np.exp(-beta * dE):
            new_tour[i:j+1] = seg[::-1]
        else:
            dE = 0

    return dE, new_tour


#@jit
def rand_reverse_no_e(tour, cities):
    a = np.random.choice(tour)
    b = np.random.choice(tour)
    i = min(np.where(tour == a)[0][0], np.where(tour == b)[0][0])
    j = max(np.where(tour == a)[0][0], np.where(tour == b)[0][0])

    if i == j:
        return 0, tour

    if j+1 == len(tour):
        j = -1

    seg = tour[i:j+1]
    if len(seg) == len(tour) or len(seg) == 0 or len(seg) == 1:
        return 0, tour

    e_old = dist(cities[tour[i-1]], cities[tour[i]]) \
            + dist(cities[tour[j]], cities[tour[j+1]])

    e_new = dist(cities[tour[i-1]], cities[tour[j]]) \
            + dist(cities[tour[i]], cities[tour[j+1]])

    dE = e_new - e_old

    tour[i:j+1] = seg[::-1]

    return dE, tour


#@jit
def temp_loop(old_tour, cities, beta, steps, tour_len_list):
    # calculate initial energy
    E = tot_energy(old_tour, cities)
    # add up energy changes
    for i in range(steps):
        dE, new_tour = rand_reverse(old_tour, cities, beta)
        E += dE
        old_tour = new_tour
        # save tour every n**2 steps
        if i % len(tour)**2 == 0:
            tour_len_list.append(E)
    return E, new_tour, tour_len_list


#@jit
def get_T0(tour, cities):
    # just do a lot of swaps and see which is largest, giving an initial temperature
    E = 0
    E_max = 0
    for i in range(100):
        dE, tour = rand_reverse_no_e(tour, cities)
        E += dE
        if E > E_max:
            E_max = E

    return E_max


#@jit
def main(tour, cities, n, t_steps):
    old_tour = tour
    #plot_path(cities, old_tour)
    T0 = 0.01 * get_T0(old_tour, cities)
    print(f"T0 = {T0}")
    T = T0
    T_list = []
    tour_list = []
    for i in range(n):
        # set temperature
        T -= T0/n - 1e-9
        # call temploop
        beta = 1/T
        E_sum, new_tour, tour_list = temp_loop(old_tour, cities, beta, t_steps, tour_list)
        #plot_path(cities, new_tour)

        T_list.append(T)

        print(f"T: {T} \t \t \t E: {E_sum}")
        print(f"{tot_energy(new_tour, cities)}")
        old_tour = new_tour

    #plot_path(cities, new_tour)
    plt.plot(tour_list)
    plt.show()

    return T_list, tour

# Cities (x, y coordinates)


cities = read_problem_input("tsp_problems/oliver30.tsp")
tour = list(range(len(cities)))

t0 = time.perf_counter()
T_list, tour = main(tour, cities, 20, len(tour)**3)
t1 = time.perf_counter()

print(f"time={t1-t0}")