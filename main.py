import random
import time
from pathlib import Path
from typing import List, Tuple, Optional
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
                try:
                    t, x, y, *_ = line.split()
                except ValueError:
                    x, y, *_ = line.split()
                cities.append((float(x.strip()), float(y.strip())))
    return cities


def read_solution_input(file_name: str) -> List[int]:
    solution_path = []
    with open(file_name, "r") as f:
        for line in f:
            line = line.strip()
            if line == "":
                continue
            if line[0].isdigit():
                # Subtract 1 to get from 1-indexed to 0-indexed cities.
                city_index = int(line.strip()) - 1
                solution_path.append(city_index)
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


def select_next_city(tabu_list: List[int], cities: List[Tuple[float, float]], trails: np.ndarray, distances:
np.ndarray, alpha: float, beta: float) -> Optional[int]:
    """
    Selects the next city to visit.
    """
    assert (len(tabu_list) != 0)
    n_cities = len(cities)
    p = np.zeros(n_cities)

    ant_position = tabu_list[-1]
    for city_i in range(n_cities):
        if city_i in tabu_list:
            continue
        p[city_i] = trails[ant_position, city_i]**alpha * 1/distances[ant_position, city_i]**beta
    assert (np.all(p >= 0))  # Probabilities should be non-negative and not nan.
    if np.all(p == 0):  # No valid next city found.
        return None
    p_cum = np.cumsum(p)
    return random.choices(range(n_cities), cum_weights=p_cum)[0]


def find_shortest_path(cities: List[Tuple[float, float]], max_iterations: int, alpha: float, beta: float, gather_stats: bool = False) -> Tuple[List[int], float]:
    """
    Finds the shortest path between cities using the Ant colony optimization algorithm.
    """
    n_cities = len(cities)
    n_ants = n_cities
    trails_start = 0.01
    trails = np.full((n_cities, n_cities), trails_start)
    d_trails = np.zeros((n_cities, n_cities))
    distances = calculate_distances(cities)

    # Init global shortest path
    shortest_distance = np.inf
    shortest_path = None

    if gather_stats:
        shortest_distance_per_cycle = []
        distance_std_per_cycle = []
        avg_node_branchings_per_cycle = []


    for cycle_number in range(max_iterations):

        tabu_lists = []
        d_trails[:] = 0

        # Construct one path for each ant.
        for ant_i in range(n_ants):
            tabu_list = [ant_i]
            while len(tabu_list) < n_cities:
                next_city = select_next_city(tabu_list, cities, trails, distances, alpha, beta)

                if next_city is None:
                    break
                else:
                    tabu_list.append(next_city)
            if len(tabu_list) == n_cities:
                tabu_lists.append(tabu_list)

        # Calculate tour length for each ant that completed a tour.
        tour_lengths = []
        for tabu_list in tabu_lists:
            n_segments = len(tabu_list)
            segment_lengths = [distances[tabu_list[i], tabu_list[(i + 1)%n_segments]] for i in range(n_segments)]
            tour_lengths.append(np.sum(segment_lengths))

        # Update trails.
        # TODO: Maybe make Q a function parameter.
        Q = 100
        for tabu_list, tour_length in zip(tabu_lists, tour_lengths):
            d_trails[tabu_list[:-1], tabu_list[1:]] += Q/tour_length
            d_trails[tabu_list[-1], tabu_list[0]] += Q/tour_length
            d_trails[tabu_list[1:], tabu_list[:-1]] += Q/tour_length
            d_trails[tabu_list[0], tabu_list[-1]] += Q/tour_length

        if cycle_number % 10 == 0:
            print(f'{cycle_number}: {np.min(tour_lengths) = :.2f}')

        rho = 0.5
        trails = rho*trails + d_trails

        # Set too small trails to 0, effectively removing them.
        trails[trails < 2**-4*trails_start] = 0

        if np.min(tour_lengths) < shortest_distance:
            shortest_distance = np.min(tour_lengths)
            shortest_path = tabu_lists[np.argmin(tour_lengths)]

        if gather_stats:
            shortest_distance_per_cycle.append(np.min(tour_lengths))
            distance_std_per_cycle.append(np.std(tour_lengths))
            avg_node_branching = np.count_nonzero(trails) / n_cities
            avg_node_branchings_per_cycle.append(avg_node_branching)

    if gather_stats:
        return shortest_path, trails, shortest_distance_per_cycle, distance_std_per_cycle, \
               avg_node_branchings_per_cycle
    else:
        return shortest_path, trails


def calculate_path_length(cities: List[Tuple[float, float]], path: List[int]) -> float:
    """
    Calculates the length of a path.
    """
    n_segments = len(path)
    assert (n_segments > 1)
    distances = calculate_distances(cities)
    segment_lengths = [distances[path[i], path[(i + 1)%n_segments]] for i in range(n_segments)]
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


def plot_trails(cities: List[Tuple[float, float]], trails: np.ndarray, ax=None):
    if ax is None:
        ax = plt.gca()
    xs = [c[0] for c in cities]
    ys = [c[1] for c in cities]
    trails = 2*trails/np.max(trails)
    ax.scatter(xs, ys, c="k")
    for i in range(len(trails)):
        for j in range(len(trails)):
            trail = trails[i, j]
            if trail != 0:
                x1, y1 = cities[i]
                x2, y2 = cities[j]
                ax.plot([x1, x2], [y1, y2], c="b", linewidth=trail)
    ax.axis("equal")


def main():
    cities: List[Tuple[float, float]]
    proposed_path: List[int]
    cities_file = Path("tsp_problems/oliver30.tsp")
    proposed_path_file = "tsp_problems/oliver30.opt.tour"
    cities = read_problem_input(cities_file)
    proposed_path = read_solution_input(proposed_path_file)

    params = {'max_iterations': 300,
              'alpha': 1,
              'beta': 5}
    t0 = time.perf_counter()
    shortest_path, trails = find_shortest_path(cities, **params)
    t1 = time.perf_counter()
    print()
    print(params)
    print(f'Finding shortest path took: {t1 - t0:.1f}s')

    shortest_path_length = calculate_path_length(cities, shortest_path)
    proposed_path_length = calculate_path_length(cities, proposed_path)
    print(f'{shortest_path_length = :.2f}')
    print(f'{proposed_path_length = :.2f}')

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3)
    plot_path(cities, shortest_path, ax=ax1)
    plot_path(cities, proposed_path, ax=ax2)
    plot_trails(cities, trails, ax=ax3)
    fig.suptitle(f'{cities_file.stem}, #cycles: {params["max_iterations"]}')
    ax1.set_title(f"Path found: {shortest_path_length:.2f}.")
    ax2.set_title(f"Optimal path: {proposed_path_length:.2f}.")
    nonzero_trails = np.count_nonzero(trails)//2
    all_trails = trails.size//2 - len(cities)
    ax3.set_title(f"Trails: {nonzero_trails}/{all_trails}.")
    plt.show()


if __name__ == '__main__':
    main()
