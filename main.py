import random
from copy import deepcopy
from typing import List, Tuple
import matplotlib.pyplot as plt
import numpy as np


def read_problem_input(file_name: str) -> List[Tuple[float, float]]:
    cities = []
    with open(file_name, "r") as f:
        for line in f:
            line = line.strip()
            if line == "":
                continue
            if line[0].isdigit():
                t, x, y = line.split()[0:3]
                cities.append((float(x.strip()), float(y.strip())))
    return cities


def read_solution_input(file_name: str) -> List[int]:
    solution_path = []
    with open(file_name, "r") as f:
        for line in f:
            if line[0].isdigit():
                solution_path.append(int(line.strip())-1)
    return solution_path


def calculate_distances(cities: List[Tuple[float, float]]):
    n_cities = len(cities)
    distances = np.zeros((n_cities, n_cities))
    cities = np.array(cities)

    for i, city_i in enumerate(cities):
        for j, city_j in enumerate(cities):
            if i == j:
                continue
            distances[i, j] = np.sqrt(np.sum((city_i - city_j)**2))

    return distances


def select_next_city(tabu_list: List[int], cities: List[Tuple[float, float]], trails: np.ndarray, distances: np.ndarray):
    """
    Selects the next city to visit.
    """
    assert(len(tabu_list) != 0)
    n_cities = len(cities)
    p = np.zeros(n_cities)

    # TODO: Maybe convert to function parameters.
    alpha = 1
    beta = 5
    ant_position = tabu_list[-1]
    for city_i in range(n_cities):
        if city_i in tabu_list:
            continue
        p[city_i] = trails[ant_position, city_i]**alpha * 1/distances[ant_position, city_i]**beta
    assert(np.all(p >= 0))  # Probabilities should be non-negative and not nan.
    assert(np.any(p > 0))
    p /= np.sum(p)
    return random.choices(list(range(n_cities)), weights=p)[0]


def find_shortest_path(cities: List[Tuple[float, float]], max_iterations: int, n_ants: int) -> List[int]:
    """
    Finds the shortest path between cities using the Ant colony optimization algorithm.
    """
    n_cities = len(cities)
    trails_min = 0.01
    trails = np.full((n_cities, n_cities), trails_min)
    d_trails = np.zeros((n_cities, n_cities))
    distances = calculate_distances(cities)

    # TODO: Check out different initializations of starting positions.
    tabu_lists_original = [[random.choice(range(n_cities))] for _ in range(n_ants)]

    # Init global shortest path
    shortest_path_global = np.inf

    for cycle_number in range(max_iterations):

        # TODO: Test if random choice of starting positionts in each iteration makes a difference.
        tabu_lists = deepcopy(tabu_lists_original)
        d_trails[:] = 0

        # Construct one path for each ant.
        for ant_i in range(n_ants):
            tabu_list = tabu_lists[ant_i]
            while len(tabu_list) < n_cities:
                next_city = select_next_city(tabu_list, cities, trails, distances)
                tabu_list.append(next_city)

        # Calculate tour length for each ant.
        tour_lengths = []
        for ant_i in range(n_ants):
            tabu_list = tabu_lists[ant_i]
            segment_lengths = [distances[tabu_list[i], tabu_list[i + 1]] for i in range(len(tabu_list) - 1)]
            tour_lengths.append(np.sum(segment_lengths))

        # Update trails.
        # TODO: Maybe make Q a function parameter.
        Q = 100
        for ant_i in range(n_ants):
            tabu_list = tabu_lists[ant_i]
            d_trails[tabu_list[:-1], tabu_list[1:]] += Q / tour_lengths[ant_i]
            d_trails[tabu_list[-1], tabu_list[0]] += Q / tour_lengths[ant_i]
            d_trails[tabu_list[1:], tabu_list[:-1]] += Q / tour_lengths[ant_i]
            d_trails[tabu_list[0], tabu_list[-1]] += Q / tour_lengths[ant_i]

        print(f'{np.min(tour_lengths) = }')
        rho = 0.5
        trails = rho*trails + d_trails
        # Limit minimum value of trails to avoid probabilities of zero.
        trails[trails < trails_min] = trails_min

        if np.any(np.min(tour_lengths) < shortest_path_global):
            shortest_path_global = tabu_lists[np.argmin(tour_lengths)]

    return shortest_path_global


def calculate_path_length(cities: List[Tuple[float, float]], path: List[int]) -> float:
    """
    Calculates the length of a path.
    """
    assert(len(path) > 1)
    distances = calculate_distances(cities)
    segment_lengths = [distances[path[i], path[i + 1]] for i in range(len(path) - 1)]
    return np.sum(segment_lengths)


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


def main():
    cities: List[Tuple[float, float]]
    proposed_path: List[int]
    cities_file = "tsp_problems/berlin52.tsp"
    proposed_path_file = "tsp_problems/berlin52.opt.tour"
    cities = read_problem_input(cities_file)
    proposed_path = read_solution_input(proposed_path_file)
    #proposed_path = list(range(14))

    max_iterations = 100
    n_ants = len(cities)
    shortest_path: List[int] = find_shortest_path(cities, max_iterations, n_ants)

    print(f'{calculate_path_length(cities, shortest_path)}')
    print(f'{calculate_path_length(cities, proposed_path)}')

    fig, (ax1, ax2) = plt.subplots(1, 2)
    plot_path(cities, shortest_path, ax=ax1)
    plot_path(cities, proposed_path, ax=ax2)
    plt.show()


if __name__ == '__main__':
    main()
