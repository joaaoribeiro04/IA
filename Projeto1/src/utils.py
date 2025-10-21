import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors


def visualize_warehouse(env, search):
    background_color = "#222222"
    text_color = "#FFFFFF"
    color1 = "#000000"
    color2 = "#282828"
    color3 = "#000000"
    path_colors = [
        "#98c379",
        "#e5c07b",
        "#e87966",
        "#61afef",
        "#56b6c2",
    ]

    border_color = background_color

    cmap = mcolors.ListedColormap(
        [background_color, color2, color3, color1] + path_colors
    )
    bounds = [-4, -3, -2, -1, 1, 2, 3, 4, 5, 6]
    norm = mcolors.BoundaryNorm(bounds, cmap.N)

    grid = np.full((env.rows, env.cols), -4, dtype=int)
    grid[env.map == 0] = -3
    grid[env.map == -1] = -1

    for pos in env.robot_positions:
        grid[pos] = 1
    for pos in env.package_positions:
        grid[pos] = 2

    fig, ax = plt.subplots(figsize=(10, 10))
    fig.patch.set_facecolor(background_color)

    ax.imshow(grid, cmap=cmap, norm=norm)
    ax.set_title("Robot Path in Environment", color=text_color)
    ax.set_xlabel("Columns", color=text_color)
    ax.set_ylabel("Rows", color=text_color)
    ax.grid(True, which="both", color=color2, linestyle="-", linewidth=1)

    package_order = search.solve_tsp(env.package_positions)
    for i, pos in enumerate(package_order, start=1):
        ax.text(
            pos[1],
            pos[0],
            str(i),
            ha="center",
            va="center",
            color=text_color,
            fontsize=12,
        )

    if hasattr(search, "path") and search.path:
        package_order = search.solve_tsp(env.package_positions)
        path_to_package_color_map = {
            package_order[i]: path_colors[i % len(path_colors)]
            for i in range(len(package_order))
        }

        current_color = path_to_package_color_map[package_order[0]]
        last_point = search.path[0]
        for point in search.path[1:]:
            if point in package_order:
                # Change color after reaching a package
                current_color = path_to_package_color_map[point]
            if is_adjacent(last_point, point):
                ax.plot(
                    [last_point[1], point[1]],
                    [last_point[0], point[0]],
                    color=current_color,
                    linewidth=2,
                )
            last_point = point

    ax.set_facecolor(background_color)
    ax.tick_params(colors=text_color, which="both")
    for spine in ax.spines.values():
        spine.set_color(border_color)

    plt.show()


def visualize_working_path(env, paths):
    background_color = "#222222"
    text_color = "#FFFFFF"
    color1 = "#000000"
    color2 = "#282828"
    color3 = "#000000"
    path_colors = [
        "#98c379",
        "#e5c07b",
        "#e87966",
        "#61afef",
        "#56b6c2",
    ]

    border_color = background_color
    cmap = mcolors.ListedColormap(
        [background_color, color2, color3, color1] + path_colors
    )

    bounds = [-4, -3, -2, -1, 1, 2, 3, 4, 5, 6]
    norm = mcolors.BoundaryNorm(bounds, cmap.N)
    grid = np.full((env.rows, env.cols), -4, dtype=int)
    grid[env.map == 0] = -3  # Free spaces
    grid[env.map == -1] = -1  # Obstacles

    for pos in env.robot_positions:
        grid[pos] = 1

    for pos in env.package_positions:
        grid[pos] = 2

    fig, ax = plt.subplots(figsize=(10, 10))
    fig.patch.set_facecolor(background_color)

    ax.imshow(grid, cmap=cmap, norm=norm)
    ax.set_title("Robot Path in Environment", color=text_color)
    ax.set_xlabel("Columns", color=text_color)
    ax.set_ylabel("Rows", color=text_color)
    ax.grid(True, which="both", color=color2, linestyle="-", linewidth=1)

    for idx, path in enumerate(paths):
        current_color = path_colors[idx % len(path_colors)]

        for i in range(len(path) - 1):
            if is_adjacent(path[i], path[i + 1]):
                ax.plot(
                    [path[i][1], path[i + 1][1]],
                    [path[i][0], path[i + 1][0]],
                    color=current_color,
                    linewidth=2,
                )

    ax.set_facecolor(background_color)
    ax.tick_params(colors=text_color, which="both")

    for spine in ax.spines.values():
        spine.set_color(border_color)

    plt.show()


def is_adjacent(point1, point2):
    """Check if two points are adjacent (horizontally or vertically)"""
    return (abs(point1[0] - point2[0]) == 1 and point1[1] == point2[1]) or (
        point1[0] == point2[0] and abs(point1[1] - point2[1]) == 1
    )


def visualize_data(absolute_data, relative_data):
    fig, ax = plt.subplots(figsize=(12, 10))
    absolute_averages = {alg: np.mean(times) for alg, times in absolute_data.items()}
    relative_averages = {alg: np.mean(times) for alg, times in relative_data.items()}

    labels = list(absolute_data.keys())
    x = np.arange(len(labels))
    absolute_averages_list = [absolute_averages[alg] for alg in labels]

    bars = ax.bar(x, absolute_averages_list, color=["blue", "green", "red", "purple"])

    ax.set_xlabel("Algorithm")
    ax.set_ylabel("Average Total Search Time (Seconds)")
    ax.set_title("Average Performance of Algorithms Over Iterations")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)

    # Adding both absolute and relative values on top of the bars
    for bar in bars:
        absolute_height = bar.get_height()
        alg = labels[bars.index(bar)]
        relative_height = relative_averages[alg]
        ax.annotate(
            f"{absolute_height:.2f}s\n({relative_height:.2f}%)",
            xy=(bar.get_x() + bar.get_width() / 2, absolute_height),
            xytext=(0, 3),  # 3 points vertical offset
            textcoords="offset points",
            ha="center",
            va="bottom",
        )

    plt.show()
