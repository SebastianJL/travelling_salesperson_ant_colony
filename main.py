from typing import List, Tuple
import matplotlib.pyplot as plt


def read_problem_input(file_name: str) -> List[Tuple[float, float]]:
    # TODO: Check if correct.

    cities = []
    with open(file_name, "r") as f:
        for line in f:
            x, y = line.split()
            cities.append((float(x), float(y)))
    return cities


def read_solution_input(file_name: str) -> List[int]:
    # TODO: Check if correct.

    with open(file_name, "r") as f:
        for line in f:
            return [int(x) for x in line.split()]


def find_shortest_path(cities: List[Tuple[float, float]], max_iterations: int, n_ants: int) -> List[int]:
    """
    Finds the shortest path between cities using the Ant colony optimization algorithm.
    """
    # TODO: Implement.
    pass


def plot_path(cities: List[Tuple[float, float]], path: List[int], ax=None):
    # TODO: Check if correct.
    if ax is None:
        ax = plt.gca()
    xs = [c[0] for c in cities]
    ys = [c[1] for c in cities]
    ax.scatter(xs, ys, c="k")
    for i in range(len(path) - 1):
        x1, y1 = cities[path[i]]
        x2, y2 = cities[path[i + 1]]
        ax.plot([x1, x2], [y1, y2], c="r")
    plt.show()


def main():
    print("Hello World!")

    cities: List[Tuple[float, float]]
    proposed_path: List[int]
    cities_file = "cities.txt"
    proposed_path_file = "proposed_path.txt"
    cities = read_problem_input(cities_file)
    proposed_path = read_solution_input(proposed_path_file)

    max_iterations = 100
    shortest_path: List[int] = find_shortest_path(cities, max_iterations, n_ants)

    fig, (ax1, ax2) = plt.subplots(1, 2)
    plot_path(cities, shortest_path, ax=ax1)
    plot_path(cities, proposed_path, ax=ax2)
    plt.show()

if __name__ == '__main__':
    main()
