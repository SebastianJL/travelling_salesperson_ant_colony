from typing import List, Tuple
import matplotlib.pyplot as plt


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


def find_shortest_path(cities: List[Tuple[float, float]], max_iterations: int, n_ants: int) -> List[int]:
    """
    Finds the shortest path between cities using the Ant colony optimization algorithm.
    """
    # TODO: Implement.
    pass


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
    plt.show()


def main():
    print("Hello World!")

    cities: List[Tuple[float, float]]
    proposed_path: List[int]
    cities_file = "tsp_problems/berlin52.tsp"
    proposed_path_file = "tsp_problems/berlin52.opt.tour"
    cities = read_problem_input(cities_file)
    print(cities)
    proposed_path = read_solution_input(proposed_path_file)

    #max_iterations = 100
    #shortest_path: List[int] = find_shortest_path(cities, max_iterations, n_ants)

    fig, (ax1, ax2) = plt.subplots(1, 2)
    #plot_path(cities, shortest_path, ax=ax1)
    plot_path(cities, proposed_path, ax=ax2)
    plt.show()


if __name__ == '__main__':
    main()
