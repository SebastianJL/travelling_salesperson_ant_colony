import random
from copy import deepcopy
from typing import List, Tuple
import matplotlib.pyplot as plt
import numpy as np


def read_problem_input(file_name: str) -> List[Tuple[float, float]]:
    cities = []
    with open(file_name, "r") as f:
        for line in f:
            if line[0].isdigit():
                t, x, y = line.split(" ")[0:3]
                cities.append((float(x.strip()), float(y.strip())))
    return cities


def read_solution_input(file_name: str) -> List[int]:
    solution_path = []
    with open(file_name, "r") as f:
        for line in f:
            if line[0].isdigit():
                solution_path.append(int(line.strip()))
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
    n_cities = len(cities)
    p = np.zeros(n_cities)

    # TODO: Maybe convert to function parameters.
    alpha = 1
    beta = 1
    for city_i in range(n_cities):
        if city_i in tabu_list:
            continue
        ant_position = tabu_list[-1]
        p[city_i] = trails[ant_position, city_i]**alpha*distances[ant_position, city_i]**beta
    p /= np.sum(p)
    assert(np.all(p >= 0))  # Probabilities should be non-negative and not nan.
    return random.choices(list(range(n_cities)), weights=list(p))[0]


def find_shortest_path(cities: List[Tuple[float, float]], max_iterations: int, n_ants: int) -> List[int]:
    """
    Finds the shortest path between cities using the Ant colony optimization algorithm.
    """
    n_cities = len(cities)
    trails = np.full((n_cities, n_cities), 1 / n_cities)
    d_trails = np.zeros((n_cities, n_cities))
    distances = calculate_distances(cities)

    # TODO: Check out diffrent initializations of starting positions.
    tabu_lists_original = [[random.choice(range(n_cities))] for _ in range(n_ants)]

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

        print(f'{np.min(tour_lengths) = }')
        rho = 0.5
        trails = rho*trails + d_trails

    return tabu_lists[np.argmin(tour_lengths)]


def plot_path(cities: List[Tuple[float, float]], path: List[int], ax=None):
    if ax is None:
        ax = plt.gca()
    xs = [c[0] for c in cities]
    ys = [c[1] for c in cities]
    ax.scatter(xs, ys, c="k")
    for i in range(len(path)):
        x1, y1 = cities[path[i] - 1]
        x2, y2 = cities[path[(i + 1)%len(path)] - 1]
        ax.plot([x1, x2], [y1, y2], c="r")


def main():
    cities: List[Tuple[float, float]]
    proposed_path: List[int]
    cities_file = "tsp_problems/berlin52.tsp"
    proposed_path_file = "tsp_problems/berlin52.opt.tour"
    cities = read_problem_input(cities_file)
    print(cities)
    proposed_path = read_solution_input(proposed_path_file)

    max_iterations = 100
    n_ants = 10
    shortest_path: List[int] = find_shortest_path(cities, max_iterations, n_ants)

    fig, (ax1, ax2) = plt.subplots(1, 2)
    plot_path(cities, shortest_path, ax=ax1)
    plot_path(cities, proposed_path, ax=ax2)
    plt.show()


if __name__ == '__main__':
    main()
