from enum import Enum
from typing import Optional
import numpy as np
import random

from .patterns import patterns


class WarehouseTypeEnum(Enum):
    Regular = 1
    Random = 2


class Environment:
    """
    Represents the environment within which robots operate in a warehouse.

    The environment can be of two types - Regular or Random. The Regular environment is created 
    based on predefined patterns, while the Random environment places obstacles randomly.

    Attributes:
        w_type (WarehouseTypeEnum): The type of warehouse environment (Regular or Random).
        rows (int): The number of rows in the warehouse.
        cols (int): The number of columns in the warehouse.
        obstacles (Optional[int]): The number of obstacles to be placed in a Random environment.
        n_packages (int): The number of packages in the warehouse.
        n_robots (int): The number of robots operating in the warehouse.
        map (np.ndarray): A 2D numpy array representing the warehouse map.
        robot_positions (list): A list of tuples representing the positions of the robots.
        package_positions (list): A list of tuples representing the positions of the packages.
        save_fn (Optional[str]): Filename to save the environment map.

    Methods:
        _place_robots():
            Randomly places robots in the warehouse.

        _place_packages():
            Randomly places packages in the warehouse.

        _create_random():
            Creates a random environment with randomly placed obstacles.

        _create_with_patterns():
            Creates a regular environment using predefined patterns.

        _save_to_file(filename: str):
            Saves the current environment map to a file.

        load_from_file(filename: str):
            Static method to load an environment from a file.

    Raises:
        ValueError: If an unknown environment type is specified.
    """

    def __init__(
        self,
        w_type: WarehouseTypeEnum,
        rows=10,
        cols=10,
        obstacle_count: Optional[int] = None,
        n_packages: int = 1,
        n_robots: int = 1,
        save_fn: Optional[str] = None,
    ):

        self.cols = cols
        self.rows = rows
        self.obstacles = obstacle_count

        self.map = np.zeros((self.rows, self.cols), dtype=np.int8)

        self.n_packages = n_packages
        self.n_robots = n_robots

        self.robot_positions = []
        self.package_positions = []

        if w_type is WarehouseTypeEnum.Regular:
            self._create_with_patterns()
        elif w_type is WarehouseTypeEnum.Random:
            self._create_random()
        else:
            raise ValueError(f"Unknown Environment type: {w_type}")

        self._place_robots()
        self._place_packages()

        if save_fn:
            self._save_to_file(save_fn)

    @staticmethod
    def load_from_file(filename):
        with open(filename, "r") as file:
            map_data = np.loadtxt(file, delimiter=",", dtype=np.int8)

        env = Environment(WarehouseTypeEnum.Regular, *map_data.shape)
        env.map = map_data
        env.rows, env.cols = map_data.shape

        env._place_robots()
        env._place_packages()

        return env

    def _place_robots(self):
        self.robot_positions = []
        free_spaces = np.argwhere(self.map == 0)
        chosen_positions = random.sample(list(free_spaces), self.n_robots)

        for pos in chosen_positions:
            self.robot_positions.append(tuple(pos))

    def _place_packages(self):
        self.package_positions = []
        free_spaces = np.argwhere(self.map == 0)
        chosen_positions = random.sample(list(free_spaces), self.n_packages)

        for pos in chosen_positions:
            self.package_positions.append(tuple(pos))

    def _create_random(self):

        def is_space_free(row, col, size):
            if col + size > self.cols:
                return False
            for i in range(size):
                if self.map[row][col + i] != 0:
                    return False
            return True

        def place_obstacle(row, col, size):
            for i in range(size):
                self.map[row][col + i] = -1

        obstacle_sizes = [
            1,
            int(self.rows / 6.0),
            int(self.rows / 8.0),
            int(self.rows / 16.0),
            int(self.rows / 32.0),
        ]

        for _ in range(self.obstacles):
            obstacle_placed = False
            while not obstacle_placed:
                r, c = random.randint(0, self.rows - 1), random.randint(
                    0, self.cols - 1
                )
                size = random.choice(obstacle_sizes)

                if is_space_free(r, c, size):
                    place_obstacle(r, c, size)
                    obstacle_placed = True

    def _create_with_patterns(self):

        def is_edge(i, j, rows, cols):
            return i == 0 or j == 0 or i + 10 == rows or j + 10 == cols

        for i in range(0, self.rows, 10):
            for j in range(0, self.cols, 10):
                # Choose a pattern. At the edge, first 2 pattern types
                # are avoided to ensure paths are not blocked
                if is_edge(i, j, self.rows, self.cols):
                    selected_pattern = random.choice(patterns[2:])
                else:
                    selected_pattern = random.choice(patterns)

                # Ensure we don't go out of bounds
                end_i = min(i + 10, self.rows)
                end_j = min(j + 10, self.cols)

                self.map[i:end_i, j:end_j] = selected_pattern[: end_i - i, : end_j - j]

    def _save_to_file(self, filename):
        np.savetxt("envs/" + filename, self.map, fmt="%d", delimiter=",")
