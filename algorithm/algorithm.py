import heapq
from collections import deque
import heapq

#Wszystkie zaimplementowane algorytmy

def dijkstra(graph, start_node, goal_node):
    # Initialize distances and priority queue
    distances = {node: float('inf') for node in graph.nodes()}
    distances[start_node] = 0
    queue = [(0, start_node)]
    came_from = {start_node: None}

    while queue:
        current_distance, current_node = heapq.heappop(queue)

        if current_distance > distances[current_node]:
            continue

        # Check neighbors
        for neighbor in graph.neighbors(current_node):
            edge_data = graph.get_edge_data(current_node, neighbor)
            weight = edge_data.get('weight', 1)  # Default weight is 1 if none is specified
            distance = current_distance + weight

            if distance < distances[neighbor]:
                distances[neighbor] = distance
                came_from[neighbor] = current_node
                heapq.heappush(queue, (distance, neighbor))

    # Reconstruct the path to the goal node
    path = reconstruct_path(came_from, goal_node) if goal_node in came_from else None

    # Clean up the path by removing None values
    cleaned_path = [p for p in path if p is not None] if path else None

    # Total weight of the path is the distance to the goal node
    total_weight = distances[goal_node] if cleaned_path else None

    return cleaned_path, total_weight


def bfs(graph, start, goal_node):
    visited = set()
    queue = deque([(start, 0)])  # Queue stores tuples (node, total_weight)
    visited.add(start)
    came_from = {start: None}
    total_weights = {start: 0}  # Store the cumulative weight to reach each node

    while queue:
        node, current_weight = queue.popleft()

        if node == goal_node:  # Stop if goal is reached
            break

        for neighbor in graph.neighbors(node):
            if neighbor not in visited:
                edge_data = graph.get_edge_data(node, neighbor)
                weight = edge_data.get('weight', 1)  # Default weight is 1 if not specified
                total_weight_to_neighbor = current_weight + weight
                
                visited.add(neighbor)
                queue.append((neighbor, total_weight_to_neighbor))
                came_from[neighbor] = node
                total_weights[neighbor] = total_weight_to_neighbor

    # Reconstruct the path
    path = reconstruct_path(came_from, goal_node) if goal_node in came_from else None
    cleaned_path = [p for p in path if p is not None] if path else None

    # The total weight of the path to the goal node
    total_weight = total_weights[goal_node] if cleaned_path else None

    return cleaned_path, total_weight


def dfs(graph, start, goal_node):
    visited = set()
    stack = [(start, 0)]  # Stack stores tuples (node, total_weight)
    came_from = {start: None}
    total_weights = {start: 0}  # Dictionary to store total weight to reach each node

    while stack:
        node, current_weight = stack.pop()

        if node == goal_node:  # Stop if the goal is reached
            break

        if node not in visited:
            visited.add(node)
            
            # Explore neighbors
            for neighbor in graph.neighbors(node):
                if neighbor not in visited:
                    edge_data = graph.get_edge_data(node, neighbor)
                    weight = edge_data.get('weight', 1)  # Default weight is 1 if none is provided
                    total_weight_to_neighbor = current_weight + weight

                    stack.append((neighbor, total_weight_to_neighbor))
                    came_from[neighbor] = node
                    total_weights[neighbor] = total_weight_to_neighbor

    # Reconstruct the path
    path = reconstruct_path(came_from, goal_node) if goal_node in came_from else None
    cleaned_path = [p for p in path if p is not None] if path else None

    # Total weight of the path to the goal node
    total_weight = total_weights[goal_node] if cleaned_path else None

    return cleaned_path, total_weight


def heuristic(node, goal):
    return abs(node - goal)  # Jeśli węzły są liczbami


def a_star(graph, start_node, goal_node):
    g_score = {node: float('inf') for node in graph.nodes()}
    g_score[start_node] = 0
    f_score = {node: float('inf') for node in graph.nodes()}
    f_score[start_node] = heuristic(start_node, goal_node)
    
    open_set = [(f_score[start_node], start_node)]
    came_from = {}

    while open_set:
        _, current = heapq.heappop(open_set)

        if current == goal_node:
            path = reconstruct_path(came_from, current)

            cleaned_path = [p for p in path if p is not None]

            total_weight = g_score[goal_node]  # Total weight of the path

            return cleaned_path, total_weight  # Return both the path and its total weight

        for neighbor in graph.neighbors(current):
            weight = graph[current][neighbor].get('weight', 1)  # Default weight is 1 if not provided
            tentative_g_score = g_score[current] + weight

            if tentative_g_score < g_score[neighbor]:
                came_from[neighbor] = current
                g_score[neighbor] = tentative_g_score
                f_score[neighbor] = g_score[neighbor] + heuristic(neighbor, goal_node)

                if neighbor not in [i[1] for i in open_set]:
                    heapq.heappush(open_set, (f_score[neighbor], neighbor))

    return None, None  # Return None if no path is found


def reconstruct_path(came_from, current):
    total_path = [current]
    while current in came_from:
        current = came_from[current]
        total_path.insert(0, current)
    return total_path