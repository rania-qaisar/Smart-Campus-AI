"""
=============================================================================
Module  : Search & Navigation Module
File    : search_module.py
Course  : AL2002 – Artificial Intelligence Lab (Spring 2026)
Project : Smart Campus AI Decision Support and Automation System
=============================================================================
Description:
    This module implements the campus route-finding engine.
    It contains ALL required search algorithms from the lab sequence and
    applies the correct operational policy:

        Graph type       → Algorithm used
        ─────────────────────────────────
        Unweighted       → BFS  (optimal for hop-count)
        Weighted + h(n)  → A*   (optimal and efficient)
        Weighted, no h   → UCS  (optimal cost, no heuristic)

    Algorithms implemented:
        Operational:
            BFS, UCS, A*
        Academic comparison:
            DFS, DLS (Depth-Limited Search), IDS (Iterative Deepening),
            Bidirectional BFS, Greedy Best-First, RBFS

    Campus Graph:
        Nodes: 13 campus locations
        Unweighted: all edges weight 1
        Weighted: edges have realistic walking-distance costs
        Coordinates: (x, y) for heuristic estimation (Euclidean)

    Pipeline position: CSP  →  [Search]  →  Final Response
=============================================================================
"""

import math
from collections import deque

# ---------------------------------------------------------------------------
# Campus Graph — Weighted (edge costs represent walking effort/time units)
# ---------------------------------------------------------------------------

WEIGHTED_GRAPH = {
    # Edges read directly from Campus_Weighted_Graph.png
    # Coordinates shown as (x,y) next to each node in the diagram
    "Main_Gate"       : {"Admin_Block": 4, "Parking": 2, "Bus_Stop": 1},
    "Parking"         : {"Main_Gate": 2, "Hostel": 5, "Cafeteria": 3},
    "Admin_Block"     : {"Main_Gate": 4, "Student_Services": 1},
    "Student_Services": {"Admin_Block": 1, "Exam_Hall": 1, "Library": 2},
    "Exam_Hall"       : {"Student_Services": 1, "Seminar_Room": 1},
    "Seminar_Room"    : {"Exam_Hall": 1, "AI_Lab": 2},
    "AI_Lab"          : {"Seminar_Room": 2, "Science_Block": 1, "Library": 3},
    "Science_Block"   : {"AI_Lab": 1, "Library": 3, "Cafeteria": 3},
    "Library"         : {"Student_Services": 2, "AI_Lab": 3, "Science_Block": 3},
    "Cafeteria"       : {"Parking": 3, "Science_Block": 3,
                         "Hostel": 2, "Medical_Center": 2},
    "Hostel"          : {"Parking": 5, "Cafeteria": 2,
                         "Medical_Center": 3, "Bus_Stop": 2},
    "Medical_Center"  : {"Hostel": 3, "Cafeteria": 2, "Bus_Stop": 2},
    "Bus_Stop"        : {"Main_Gate": 1, "Hostel": 2, "Medical_Center": 2},
}

# Unweighted version — edges from Campus_UnWeighted_Graph.png
# All edge weights = 1 (hop count only)
UNWEIGHTED_GRAPH = {
    "Main_Gate"       : {"Bus_Stop": 1, "Admin_Block": 1, "Parking": 1,
                         "Hostel": 1},
    "Bus_Stop"        : {"Main_Gate": 1, "Medical_Center": 1, "Hostel": 1},
    "Parking"         : {"Main_Gate": 1, "Hostel": 1, "Science_Block": 1},
    "Admin_Block"     : {"Main_Gate": 1, "Student_Services": 1},
    "Student_Services": {"Admin_Block": 1, "Exam_Hall": 1, "Science_Block": 1},
    "Exam_Hall"       : {"Student_Services": 1, "Science_Block": 1,
                         "Seminar_Room": 1},
    "Seminar_Room"    : {"Exam_Hall": 1, "Science_Block": 1, "Library": 1},
    "Science_Block"   : {"Parking": 1, "Student_Services": 1, "Exam_Hall": 1,
                         "Seminar_Room": 1, "Library": 1, "AI_Lab": 1,
                         "Cafeteria": 1, "Hostel": 1},
    "Library"         : {"Seminar_Room": 1, "Science_Block": 1, "AI_Lab": 1},
    "AI_Lab"          : {"Science_Block": 1, "Library": 1, "Cafeteria": 1},
    "Cafeteria"       : {"Science_Block": 1, "AI_Lab": 1, "Hostel": 1},
    "Hostel"          : {"Main_Gate": 1, "Bus_Stop": 1, "Parking": 1,
                         "Science_Block": 1, "Cafeteria": 1,
                         "Medical_Center": 1},
    "Medical_Center"  : {"Bus_Stop": 1, "Hostel": 1},
}

# Node coordinates (x, y) — read directly from weighted graph diagram labels
# e.g. Admin_Block (3,5) means x=3, y=5
NODE_COORDS = {
    "Main_Gate"       : (0, 4),
    "Parking"         : (2, 4),
    "Admin_Block"     : (3, 5),
    "Student_Services": (6, 5),
    "Exam_Hall"       : (8, 5),
    "Seminar_Room"    : (10, 4),
    "AI_Lab"          : (9, 2),
    "Science_Block"   : (7, 1),
    "Library"         : (6, 2),
    "Cafeteria"       : (4, 1),
    "Hostel"          : (2, 0),
    "Medical_Center"  : (1, 1),
    "Bus_Stop"        : (0, 1),
}


# ---------------------------------------------------------------------------
# Heuristic function (Euclidean distance)
# ---------------------------------------------------------------------------

def heuristic(node: str, goal: str) -> float:
    """
    Euclidean distance heuristic for A* and Greedy Best-First Search.
    Uses predefined (x, y) coordinates for each campus node.

    Parameters:
        node (str): Current node identifier.
        goal (str): Goal node identifier.

    Returns:
        float: Estimated cost to reach goal from node.
    """
    x1, y1 = NODE_COORDS.get(node, (0, 0))
    x2, y2 = NODE_COORDS.get(goal, (0, 0))
    return math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)


# ---------------------------------------------------------------------------
# Path reconstruction helper
# ---------------------------------------------------------------------------

def _reconstruct(came_from: dict, start: str, goal: str) -> list:
    """
    Reconstructs the path from start to goal using the came_from dictionary.

    Parameters:
        came_from (dict): Maps each node to its predecessor.
        start     (str) : Starting node.
        goal      (str) : Goal node.

    Returns:
        list: Ordered list of nodes from start to goal, or [] if unreachable.
    """
    path = []
    node = goal
    while node != start:
        if node not in came_from:
            return []
        path.append(node)
        node = came_from[node]
    path.append(start)
    path.reverse()
    return path


# ===========================================================================
# OPERATIONAL ALGORITHMS
# ===========================================================================

def bfs(graph: dict, start: str, goal: str) -> dict:
    """
    Breadth-First Search — optimal for unweighted graphs (minimises hops).

    Algorithm:
        Uses a FIFO queue. Explores neighbours level by level.
        Guaranteed to find the shortest path (by edge count).

    Parameters:
        graph (dict): Adjacency dict {node: {neighbour: cost}}.
        start (str) : Start node.
        goal  (str) : Goal node.

    Returns:
        dict: Standard search result object.
    """
    if start == goal:
        return _make_result("BFS", [start], 0, 0, 1)

    queue      = deque([[start]])
    visited    = {start}
    nodes_exp  = 0

    while queue:
        path = queue.popleft()
        node = path[-1]
        nodes_exp += 1

        for neighbour in graph.get(node, {}):
            if neighbour == goal:
                full_path = path + [neighbour]
                cost      = sum(graph[full_path[i]][full_path[i+1]]
                                for i in range(len(full_path)-1))
                return _make_result("BFS", full_path, cost, len(full_path)-1, nodes_exp)
            if neighbour not in visited:
                visited.add(neighbour)
                queue.append(path + [neighbour])

    return _make_result("BFS", [], 0, 0, nodes_exp, found=False)


def ucs(graph: dict, start: str, goal: str) -> dict:
    """
    Uniform Cost Search — optimal for weighted graphs without heuristic.

    Algorithm:
        Uses a priority queue ordered by cumulative path cost.
        Expands lowest-cost unexplored node first.
        Equivalent to Dijkstra's algorithm for single-goal search.

    Parameters:
        graph (dict): Weighted adjacency dict.
        start (str) : Start node.
        goal  (str) : Goal node.

    Returns:
        dict: Standard search result object.
    """
    import heapq
    # (cost, node, path)
    frontier   = [(0, start, [start])]
    visited    = {}
    nodes_exp  = 0

    while frontier:
        cost, node, path = heapq.heappop(frontier)
        if node in visited:
            continue
        visited[node] = cost
        nodes_exp    += 1

        if node == goal:
            return _make_result("UCS", path, cost, len(path)-1, nodes_exp)

        for neighbour, edge_cost in graph.get(node, {}).items():
            if neighbour not in visited:
                heapq.heappush(frontier,
                               (cost + edge_cost, neighbour, path + [neighbour]))

    return _make_result("UCS", [], 0, 0, nodes_exp, found=False)


def astar(graph: dict, start: str, goal: str) -> dict:
    """
    A* Search — optimal and efficient for weighted graphs with heuristic.

    Algorithm:
        f(n) = g(n) + h(n)
        g(n) = actual cost from start to n
        h(n) = Euclidean distance heuristic to goal
        Uses a min-heap priority queue ordered by f(n).

    Parameters:
        graph (dict): Weighted adjacency dict.
        start (str) : Start node.
        goal  (str) : Goal node.

    Returns:
        dict: Standard search result object.
    """
    import heapq
    # (f, g, node, path)
    frontier   = [(heuristic(start, goal), 0, start, [start])]
    visited    = {}
    nodes_exp  = 0

    while frontier:
        f, g, node, path = heapq.heappop(frontier)
        if node in visited:
            continue
        visited[node] = g
        nodes_exp    += 1

        if node == goal:
            return _make_result("A*", path, g, len(path)-1, nodes_exp)

        for neighbour, edge_cost in graph.get(node, {}).items():
            if neighbour not in visited:
                new_g = g + edge_cost
                new_f = new_g + heuristic(neighbour, goal)
                heapq.heappush(frontier,
                               (new_f, new_g, neighbour, path + [neighbour]))

    return _make_result("A*", [], 0, 0, nodes_exp, found=False)


# ===========================================================================
# ACADEMIC / COMPARISON ALGORITHMS
# ===========================================================================

def dfs(graph: dict, start: str, goal: str) -> dict:
    """
    Depth-First Search — explores deepest path first (not optimal).

    Algorithm:
        Uses a LIFO stack. May not find the shortest path.
        Included for academic comparison.

    Parameters:
        graph, start, goal: Same as BFS.

    Returns:
        dict: Standard search result object.
    """
    stack     = [[start]]
    visited   = set()
    nodes_exp = 0

    while stack:
        path = stack.pop()
        node = path[-1]
        if node in visited:
            continue
        visited.add(node)
        nodes_exp += 1

        if node == goal:
            cost = sum(graph[path[i]][path[i+1]] for i in range(len(path)-1))
            return _make_result("DFS", path, cost, len(path)-1, nodes_exp)

        for neighbour in reversed(list(graph.get(node, {}).keys())):
            if neighbour not in visited:
                stack.append(path + [neighbour])

    return _make_result("DFS", [], 0, 0, nodes_exp, found=False)


def dls(graph: dict, start: str, goal: str, limit: int = 5) -> dict:
    """
    Depth-Limited Search — DFS with a depth cutoff.

    Prevents infinite loops in deep graphs.
    Included for academic comparison.

    Parameters:
        graph (dict): Adjacency dict.
        start (str) : Start node.
        goal  (str) : Goal node.
        limit (int) : Maximum depth to explore.

    Returns:
        dict: Standard search result object.
    """
    def _dls_recursive(path, depth):
        node = path[-1]
        if node == goal:
            return path
        if depth == 0:
            return None
        for nb in graph.get(node, {}):
            if nb not in path:
                result = _dls_recursive(path + [nb], depth - 1)
                if result is not None:
                    return result
        return None

    result = _dls_recursive([start], limit)
    if result:
        cost = sum(graph[result[i]][result[i+1]] for i in range(len(result)-1))
        return _make_result(f"DLS(limit={limit})", result, cost, len(result)-1, 0)
    return _make_result(f"DLS(limit={limit})", [], 0, 0, 0, found=False)


def ids(graph: dict, start: str, goal: str, max_depth: int = 10) -> dict:
    """
    Iterative Deepening Search — repeats DLS with increasing depth limits.

    Combines memory efficiency of DFS with optimality of BFS on unweighted graphs.
    Included for academic comparison.

    Parameters:
        graph     (dict): Adjacency dict.
        start     (str) : Start node.
        goal      (str) : Goal node.
        max_depth (int) : Maximum depth to attempt.

    Returns:
        dict: Standard search result object.
    """
    for depth in range(max_depth + 1):
        result = dls(graph, start, goal, limit=depth)
        if result["found"]:
            result["algorithm_used"] = f"IDS(depth={depth})"
            return result
    return _make_result("IDS", [], 0, 0, 0, found=False)


def bidirectional_bfs(graph: dict, start: str, goal: str) -> dict:
    """
    Bidirectional BFS — runs BFS simultaneously from start and goal.

    Meets in the middle, reducing search space significantly.
    Included for academic comparison.

    Parameters:
        graph, start, goal: Same as BFS.

    Returns:
        dict: Standard search result object.
    """
    if start == goal:
        return _make_result("Bidirectional BFS", [start], 0, 0, 1)

    front_visited = {start: [start]}
    back_visited  = {goal:  [goal]}
    front_queue   = deque([start])
    back_queue    = deque([goal])
    nodes_exp     = 0

    while front_queue and back_queue:
        # Expand forward
        node = front_queue.popleft()
        nodes_exp += 1
        for nb in graph.get(node, {}):
            if nb not in front_visited:
                front_visited[nb] = front_visited[node] + [nb]
                front_queue.append(nb)
            if nb in back_visited:
                path = front_visited[nb] + list(reversed(back_visited[nb][:-1]))
                cost = sum(graph[path[i]][path[i+1]] for i in range(len(path)-1))
                return _make_result("Bidirectional BFS", path, cost, len(path)-1, nodes_exp)

        # Expand backward
        node = back_queue.popleft()
        nodes_exp += 1
        for nb in graph.get(node, {}):
            if nb not in back_visited:
                back_visited[nb] = back_visited[node] + [nb]
                back_queue.append(nb)
            if nb in front_visited:
                path = front_visited[nb] + list(reversed(back_visited[nb][:-1]))
                cost = sum(graph[path[i]][path[i+1]] for i in range(len(path)-1))
                return _make_result("Bidirectional BFS", path, cost, len(path)-1, nodes_exp)

    return _make_result("Bidirectional BFS", [], 0, 0, nodes_exp, found=False)


def greedy_bfs(graph: dict, start: str, goal: str) -> dict:
    """
    Greedy Best-First Search — uses only heuristic h(n), ignores g(n).

    Fast but not guaranteed optimal.
    Included for academic comparison.

    Parameters:
        graph, start, goal: Same as A*.

    Returns:
        dict: Standard search result object.
    """
    import heapq
    frontier  = [(heuristic(start, goal), start, [start])]
    visited   = set()
    nodes_exp = 0

    while frontier:
        h, node, path = heapq.heappop(frontier)
        if node in visited:
            continue
        visited.add(node)
        nodes_exp += 1

        if node == goal:
            cost = sum(graph[path[i]][path[i+1]] for i in range(len(path)-1))
            return _make_result("Greedy BFS", path, cost, len(path)-1, nodes_exp)

        for nb, _ in graph.get(node, {}).items():
            if nb not in visited:
                heapq.heappush(frontier, (heuristic(nb, goal), nb, path + [nb]))

    return _make_result("Greedy BFS", [], 0, 0, nodes_exp, found=False)


def rbfs(graph: dict, start: str, goal: str) -> dict:
    """
    Recursive Best-First Search — memory-efficient alternative to A*.

    Keeps track of the best alternative path f-value and backtracks when
    the current path exceeds it.
    Included for academic comparison.

    Parameters:
        graph, start, goal: Same as A*.

    Returns:
        dict: Standard search result object.
    """
    INF = float("inf")
    nodes_exp = [0]

    def _rbfs_inner(node, path, g, f_limit):
        if node == goal:
            return path, g
        nodes_exp[0] += 1

        successors = []
        for nb, cost in graph.get(node, {}).items():
            if nb not in path:
                new_g = g + cost
                new_f = new_g + heuristic(nb, goal)
                successors.append((new_f, new_g, nb, path + [nb]))

        if not successors:
            return None, INF

        while True:
            successors.sort(key=lambda x: x[0])
            best_f, best_g, best_nb, best_path = successors[0]
            if best_f > f_limit:
                return None, best_f
            alt_f = successors[1][0] if len(successors) > 1 else INF
            result, new_f = _rbfs_inner(best_nb, best_path, best_g,
                                         min(f_limit, alt_f))
            successors[0] = (new_f, best_g, best_nb, best_path)
            if result is not None:
                return result, new_f

    result, _ = _rbfs_inner(start, [start], 0, float("inf"))
    if result:
        cost = sum(graph[result[i]][result[i+1]] for i in range(len(result)-1))
        return _make_result("RBFS", result, cost, len(result)-1, nodes_exp[0])
    return _make_result("RBFS", [], 0, 0, nodes_exp[0], found=False)


# ---------------------------------------------------------------------------
# Standard result object builder
# ---------------------------------------------------------------------------

def _make_result(algorithm: str, path: list, cost, steps: int,
                 nodes_exp: int, found: bool = True) -> dict:
    """
    Builds a standard search result dictionary.

    Parameters:
        algorithm  (str) : Name of algorithm used.
        path       (list): Sequence of nodes from start to goal.
        cost       (int/float): Total path cost.
        steps      (int) : Number of edges traversed.
        nodes_exp  (int) : Number of nodes expanded during search.
        found      (bool): Whether a path was found.

    Returns:
        dict: Standard search result object.
    """
    return {
        "algorithm_used": algorithm,
        "path"          : path,
        "cost"          : round(cost, 2) if path else 0,
        "steps"         : steps,
        "nodes_expanded": nodes_exp,
        "found"         : found,
    }


# ===========================================================================
# Operational policy selector
# ===========================================================================

def select_algorithm(weighted: bool = True) -> str:
    """
    Applies the project's operational search policy to select an algorithm.

        Unweighted graph      → BFS
        Weighted + heuristic  → A*
        Weighted, no heuristic → UCS (fallback)

    Parameters:
        weighted (bool): True if using weighted graph (default True).

    Returns:
        str: Algorithm name ("BFS", "A*", or "UCS").
    """
    if not weighted:
        return "BFS"
    return "A*"  # Heuristic (Euclidean) is always available in this project


# ===========================================================================
# Main entry point for the pipeline
# ===========================================================================

def run_search(source: str, destination: str,
               weighted: bool = True,
               comparison_mode: bool = False) -> dict:
    """
    Main entry point for the Search & Navigation module.

    Selects the appropriate algorithm according to project policy and
    computes the campus route. Optionally runs all algorithms for
    academic comparison mode.

    Parameters:
        source          (str) : Starting campus location.
        destination     (str) : Target campus location.
        weighted        (bool): Use weighted graph (default True → A*).
        comparison_mode (bool): If True, run all algorithms and compare.

    Returns:
        dict: {
            "algorithm_used" : str,
            "path"           : list,
            "cost"           : float,
            "steps"          : int,
            "nodes_expanded" : int,
            "found"          : bool,
            "comparison"     : dict  (only if comparison_mode=True)
        }
    """
    if not source or not destination:
        return _make_result("N/A", [], 0, 0, 0, found=False)

    graph = WEIGHTED_GRAPH if weighted else UNWEIGHTED_GRAPH

    algo = select_algorithm(weighted)

    # Operational result
    if algo == "BFS":
        result = bfs(graph, source, destination)
    elif algo == "A*":
        result = astar(graph, source, destination)
    else:
        result = ucs(graph, source, destination)

    # Academic comparison mode
    if comparison_mode:
        comparison = {}
        for name, fn in [
            ("BFS", lambda: bfs(UNWEIGHTED_GRAPH, source, destination)),
            ("DFS", lambda: dfs(graph, source, destination)),
            ("DLS", lambda: dls(graph, source, destination, limit=6)),
            ("IDS", lambda: ids(graph, source, destination)),
            ("UCS", lambda: ucs(graph, source, destination)),
            ("Bidirectional BFS", lambda: bidirectional_bfs(graph, source, destination)),
            ("Greedy BFS",        lambda: greedy_bfs(graph, source, destination)),
            ("A*",                lambda: astar(graph, source, destination)),
            ("RBFS",              lambda: rbfs(graph, source, destination)),
        ]:
            try:
                comparison[name] = fn()
            except Exception as e:
                comparison[name] = {"error": str(e)}
        result["comparison"] = comparison

    return result
