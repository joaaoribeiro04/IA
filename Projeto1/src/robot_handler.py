import time
import numpy as np
from .algorithms import Search, AlgorithmTypeEnum, HeuristicEnum
from .environment import Environment


class MultiRobotHandler:
    """
    Handles multi-robot task allocation and path planning in a specific environment.

    This class is responsible for allocating tasks to multiple robots and planning their paths
    efficiently using a combination of informed search strategies and heuristics.

    Attributes:
        env (Environment): The environment where the robots operate.
        algorithm_type (AlgorithmTypeEnum): Type of pathfinding algorithm used.
        heuristic (HeuristicEnum): Heuristic used in the pathfinding algorithm.

    Methods:
        calculate_cost(robot_position, package_position, assigned_tasks):
            Estimates the cost of assigning a package to a robot based on distance and workload.
            This method is used as a custom heuristic.

        allocate_tasks():
            Allocates tasks to robots based on the cost calculated by 'calculate_cost'.

        solve_tsp(start_position, package_locations):
            Solves the Traveling Salesman Problem (TSP) for each robot, using
            nearest neighbors.

        plan_paths():
            Plans paths for each robot, allocates tasks, and calculates the total cost.
    """

    def __init__(
        self,
        environment: Environment,
        algorithm_type: AlgorithmTypeEnum,
        heuristic: HeuristicEnum,
    ):
        self.env = environment
        self.algorithm_type = algorithm_type
        self.heuristic = heuristic

    def calculate_cost(self, robot_position, package_position, assigned_tasks):
        # Custom heuristic, uses manhattan and penalizes with amount of packages
        # assigned.
        return (
            sum(abs(a - b) for a, b in zip(robot_position, package_position)) *
            0.9
        ) + (len(assigned_tasks) * (self.env.rows * 0.01))

    def allocate_tasks(self):
        allocations = [[] for _ in range(self.env.n_robots)]
        remaining_packages = self.env.package_positions.copy()
        while remaining_packages:
            min_cost = float("inf")
            best_robot = -1
            best_package = None

            for i, robot_position in enumerate(self.env.robot_positions):
                for package in remaining_packages:
                    cost = self.calculate_cost(robot_position, package, allocations[i])
                    if cost < min_cost:
                        min_cost = cost
                        best_robot = i
                        best_package = package

            allocations[best_robot].append(best_package)
            remaining_packages.remove(best_package)

        return allocations

    def solve_tsp(self, start_position, package_locations):
        if not package_locations:
            return []

        unvisited = set(package_locations)
        path = [start_position]

        while unvisited:
            nearest = min(
                unvisited,
                key=lambda loc: np.linalg.norm(np.array(path[-1]) - np.array(loc)),
            )
            path.append(nearest)
            unvisited.remove(nearest)

        return path[1:]

    def plan_paths(self):
        start_time = time.time()

        task_allocations = self.allocate_tasks()
        robot_paths = []
        individual_working_times = []
        max_working_time = 0

        for i, robot_position in enumerate(self.env.robot_positions):
            search = Search(self.algorithm_type, self.env, self.heuristic)
            robot_path = []
            robot_working_time = 0

            tsp_route = self.solve_tsp(robot_position, task_allocations[i])

            for package_position in tsp_route:
                path, path_cost = search.algorithm(robot_position, package_position)
                robot_path.extend(path)
                robot_working_time += path_cost
                robot_position = package_position

            robot_paths.append(robot_path)
            individual_working_times.append(robot_working_time)
            max_working_time = max(max_working_time, robot_working_time)

        search_cost = time.time() - start_time
        total_cost = max_working_time + search_cost

        return robot_paths, individual_working_times, total_cost
