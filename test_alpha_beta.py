import time
from pathlib import Path

from matplotlib import pyplot as plt

from main import read_problem_input, read_solution_input, find_shortest_path, calculate_path_length


def main():
    cities_file = Path("tsp_problems/oliver30.tsp")
    proposed_path_file = "tsp_problems/oliver30.opt.tour"
    cities = read_problem_input(cities_file)
    proposed_path = read_solution_input(proposed_path_file)

    alphas = (0, 0.5, 1, 2, 5)
    betas = (0, 1, 2, 5)
    alphas = (1,)
    betas = (5,)
    for alpha in alphas:
        for beta in betas:
            params = {'max_iterations': 300,
                      'alpha'         : alpha,
                      'beta'          : beta,
                      'gather_stats'  : True,
                      }
            t0 = time.perf_counter()
            shortest_path, trails, shortest_distance_per_cycle, distance_std_per_cycle, avg_node_branchings_per_cycle\
                = find_shortest_path(cities, **params)
            t1 = time.perf_counter()
            print(params)
            print(f'\nFinding shortest path took: {t1 - t0:.1f}s')

    shortest_path_length = calculate_path_length(cities, shortest_path)
    proposed_path_length = calculate_path_length(cities, proposed_path)
    print(f'{shortest_path_length = :.2f}')
    print(f'{proposed_path_length = :.2f}')

    plt.figure()
    plt.plot(shortest_distance_per_cycle)
    plt.xlabel('Cycles')
    plt.ylabel('Best tour length')

    plt.figure()
    plt.plot(distance_std_per_cycle)
    plt.ylim((0, 80))
    plt.xlabel('Cycles')
    plt.ylabel('Tour length standard deviation')

    plt.figure()
    plt.plot(avg_node_branchings_per_cycle)
    plt.xlabel('Cycles')
    plt.ylabel('Average node branching')
    plt.show()


if __name__ == '__main__':
    main()
