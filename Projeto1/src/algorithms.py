from enum import Enum
import collections
import numpy as np
import time
import heapq
import math
from typing import Optional
from .environment import Environment

DIRECTIONS = np.array([(0, 1), (1, 0), (0, -1), (-1, 0)])


class HeuristicEnum(Enum):
    MANHATTAN = "Manhattan distance on a square grid."
    EUCLIDEAN = "Euclidian distance on a square grid"


class AlgorithmTypeEnum(Enum):
    """ """

    BFS = "Breadth First Search (BFS)"
    DIJKSTRA = "Dijkstra's Search"
    A_STAR = "A* Search"
    IDA_STAR = "Iterative Deepening A* (IDA) Search"


class Search:
    """
    Implements various search algorithms for pathfinding in a given environment.

    Supports BFS, Dijkstra's, A*, and IDA* algorithms with Manhattan or Euclidean heuristics.

    Attributes:
        env (Environment): The environment where the search is performed.
        algorithm (callable): The selected search algorithm.
        a_type (AlgorithmTypeEnum): The type of the selected algorithm.
        path (list): The path found by the search algorithm.
        total_search_time (float): The total time taken for the search and path traversal.

    Methods:
        print_cost():
            Prints the cost of the path found, search time, and other relevant details.

        solve_tsp(package_locations):
            A static method for solving the Traveling Salesman Problem using a greedy approach.

        _run():
            Runs the selected search algorithm to find a path in the environment.

        _calculate_cost(current_dir, new_dir):
            Calculates the cost of moving from one direction to another.

        _bfs_search(goal_positions, start_positions):
            Breadth First Search algorithm.

        _dijkstra_search(goal_positions, start_positions):
            Dijkstra's algorithm.

        _heuristic_manhattan(a, b):
            Calculates Manhattan distance between two points.

        _heuristic_euclidean(a, b):
            Calculates Euclidean distance between two points.

        _heuristic(a, b):
            Wrapper for the selected heuristic function.

        _a_star_search(goal_positions, start_positions):
            A* search algorithm.

        _ida_star_search(goal_positions, start_positions):
            IDA* search algorithm.

    Raises:
        ValueError: If an unknown algorithm type is specified.
    """

    def __init__(
        self,
        a_type: AlgorithmTypeEnum,
        environment: Environment,
        heuristic: Optional[HeuristicEnum] = None,
    ):
        self.env = environment
        self.algorithm = self._a_star_search
        self.a_type = a_type.value
        self.path = None

        if a_type is AlgorithmTypeEnum.BFS:
            self.algorithm = self._bfs_search

        elif a_type is AlgorithmTypeEnum.A_STAR:
            self.algorithm = self._a_star_search

        elif a_type is AlgorithmTypeEnum.DIJKSTRA:
            self.algorithm = self._dijkstra_search

        elif a_type is AlgorithmTypeEnum.IDA_STAR:
            self.algorithm = self._a_star_search

        else:
            raise ValueError(f"Unknown AlgorithmType: {a_type}")

        if heuristic is HeuristicEnum.MANHATTAN:
            self._heuristic_type = self._heuristic_manhattan
        elif heuristic is HeuristicEnum.EUCLIDEAN:
            self._heuristic_type = self._heuristic_euclidean

        self._run()

        if self.path and self.env.n_packages == 1:
            self.total_search_time = self.search_time + self.cost * 2 + 2
        elif self.path and self.env.n_packages > 1:
            self.total_search_time = self.search_time + self.cost
        else:
            print("No path is available!")

    def print_cost(self):
        if self.path:
            if self.env.n_packages == 1:
                print("Cost to Package: ", self.cost, " units of time")
                print(
                    "Solution found with cost of: ", self.cost * 2 + 2, " units of time"
                )
                print("Search Time: ", self.search_time, "seconds")
                print(
                    "Total Time: ",
                    self.search_time + self.cost * 2 + 2,
                    "seconds (assuming unit of time is second)",
                    " with ",
                    self.a_type,
                )
            else:
                print("Solution found with: ", self.cost, " cost")
                print("Search Time: ", self.search_time)
                print(
                    "Total Time: ", self.search_time + self.cost, " with ", self.a_type
                )
        else:
            raise ValueError("No path available!")

    @staticmethod
    def solve_tsp(package_locations):
        # Basic implementation using a greedy nearest-neighbor approach
        if not package_locations:
            return []

        start = package_locations[0]
        unvisited = set(package_locations)
        path = [start]
        unvisited.remove(start)

        while unvisited:
            nearest = min(
                unvisited,
                key=lambda loc: np.linalg.norm(np.array(path[-1]) - np.array(loc)),
            )
            path.append(nearest)
            unvisited.remove(nearest)

        return path

    def _run(self):
        try:
            s_t = time.time()
            total_path = []
            total_cost = 0

            if self.env.n_packages > 1:
                package_order = Search.solve_tsp(self.env.package_positions)
                current_position = self.env.robot_positions[0]

                for goal_position in package_order:
                    path, cost = self.algorithm(current_position, goal_position)
                    total_path.extend(path)
                    total_cost += cost
                    current_position = goal_position

                path, cost = self.algorithm(
                    current_position, self.env.robot_positions[0]
                )
                total_path.extend(path)
                total_cost += cost

                self.path = total_path
                self.cost = total_cost
            else:
                self.path, self.cost = self.algorithm(
                    self.env.package_positions[0], self.env.robot_positions[0]
                )
            e_t = time.time()
            self.search_time = e_t - s_t
        except Exception as e:
            print(e)

    def _calculate_cost(self, current_dir, new_dir):
        if (current_dir == new_dir).all():
            return 1.0
        elif (current_dir == -new_dir).all():
            return 3.0
        return 1.5

    def _bfs_search(self, goal_positions, start_positions):
        rows, cols = self.env.map.shape
        queue = collections.deque([(start_positions, (0, 1), 0, [])])
        visited = set()
        visited.add(start_positions)

        goal = goal_positions
        best_cost = float("inf")
        best_path = []

        while queue:
            position, direction, cost, path = queue.popleft()
            r, c = position

            if position == goal and cost < best_cost:
                best_cost = cost
                best_path = path + [position]

            for new_dir in DIRECTIONS:
                next_position = (r + new_dir[0], c + new_dir[1])
                if (
                    0 <= next_position[0] < rows
                    and 0 <= next_position[1] < cols
                    and self.env.map[next_position] != -1
                ):
                    if next_position not in visited:
                        visited.add(next_position)
                        new_cost = cost + self._calculate_cost(
                            np.array(direction), np.array(new_dir)
                        )
                        queue.append(
                            (next_position, new_dir, new_cost, path + [position])
                        )

        return best_path, best_cost

    def _dijkstra_search(self, goal_positions, start_positions):
        start = start_positions
        goal = goal_positions
        rows, cols = self.env.map.shape

        queue = []
        heapq.heappush(queue, (0, start, [(0, 1)], []))

        cost_so_far = {start: 0}

        while queue:
            current_cost, (r, c), direction, path = heapq.heappop(queue)

            if (r, c) == goal:
                return path + [(r, c)], current_cost

            for new_dir in DIRECTIONS:
                next_r, next_c = np.array([r, c]) + new_dir
                if (
                    0 <= next_r < rows
                    and 0 <= next_c < cols
                    and self.env.map[next_r, next_c] != -1
                ):
                    new_position = (next_r, next_c)
                    new_cost = current_cost + self._calculate_cost(
                        direction[-1], new_dir
                    )

                    if (
                        new_position not in cost_so_far
                        or new_cost < cost_so_far[new_position]
                    ):
                        cost_so_far[new_position] = new_cost
                        heapq.heappush(
                            queue,
                            (
                                new_cost,
                                new_position,
                                direction + [new_dir],
                                path + [(r, c)],
                            ),
                        )

    def _heuristic_manhattan(self, a, b):
        return abs(a[0] - b[0]) + abs(a[1] - b[1])

    def _heuristic_euclidean(self, a, b):
        return math.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2)

    def _heuristic(self, a, b):
        return self._heuristic_type(a, b)

    def _a_star_search(self, goal_positions, start_positions):
        start = start_positions
        goal = goal_positions
        rows, cols = self.env.map.shape
        queue = []
        heapq.heappush(
            queue, (0 + self._heuristic(start, goal), 0, start, [(0, 1)], [])
        )
        cost_so_far = {start: 0}

        while queue:
            _, current_cost, (r, c), direction, path = heapq.heappop(queue)

            if (r, c) == goal:
                return path + [(r, c)], current_cost

            for new_dir in DIRECTIONS:
                next_r, next_c = np.array([r, c]) + new_dir
                if (
                    0 <= next_r < rows
                    and 0 <= next_c < cols
                    and self.env.map[next_r, next_c] != -1
                ):
                    new_cost = current_cost + self._calculate_cost(
                        direction[-1], new_dir
                    )
                    new_position = (next_r, next_c)

                    if (
                        new_position not in cost_so_far
                        or new_cost < cost_so_far[new_position]
                    ):
                        cost_so_far[new_position] = new_cost
                        priority = new_cost + self._heuristic(new_position, goal)
                        heapq.heappush(
                            queue,
                            (
                                priority,
                                new_cost,
                                new_position,
                                direction + [new_dir],
                                path + [(r, c)],
                            ),
                        )

    def _ida_star_search(self, goal_positions, start_positions):
        start = start_positions
        goal = goal_positions

        def search(path, g, bound):
            node = path[-1]
            f = g + self._heuristic(node, goal)
            if f > bound:
                return f
            if node == goal:
                return True
            min_bound = float("inf")
            for new_dir in DIRECTIONS:
                next_r, next_c = np.array(node) + new_dir
                if (
                    0 <= next_r < self.env.map.shape[0]
                    and 0 <= next_c < self.env.map.shape[1]
                    and self.env.map[next_r, next_c] != -1
                    and (next_r, next_c) not in path
                ):
                    path.append((next_r, next_c))
                    new_cost = g + self._calculate_cost(
                        np.array(DIRECTIONS[-1]), new_dir
                    )
                    t = search(path, new_cost, bound)
                    if t is True:
                        return True
                    if t < min_bound:
                        min_bound = t
                    path.pop()
            return min_bound

        bound = self._heuristic(start, goal)
        path = [start]
        while True:
            t = search(path, 0, bound)
            if t is True:
                return path[:], sum(
                    [
                        self._calculate_cost(
                            np.array(DIRECTIONS[-1]), np.array(DIRECTIONS[i % 4])
                        )
                        for i in range(len(path) - 1)
                    ]
                )
            if t == float("inf"):
                return None, float("inf")
            bound = t
