import networkx as nx
import random
import tracemalloc
import time
import matplotlib.pyplot as plt

# generująca różnych typow grafów

generated_files = []

def generate_graphs():
    graphs = {
        "Large Sparse Graph": create_large_sparse_graph(5,10),
        "Large Dense Graph": create_dense_graph(50),
        "Large Weighted Graph": create_large_weighted_graph(20, 35),
        "Small Sparse Graph": create_small_sparse_graph(5,5),
        "Small Dense Graph": create_small_dense_graph(10),
        "Small Weighted Graph": create_weighted_graph(10, 15),
    }
    return graphs


# Funkcja do stworzenia grafu nieważonego
def create_unweighted_graph(num_nodes, num_edges):
    G = nx.Graph()
    G.add_nodes_from(range(num_nodes))
    
    # Tworzenie listy możliwych krawędzi (unikalnych par węzłów)
    possible_edges = [(i, j) for i in range(num_nodes) for j in range(i + 1, num_nodes)]
    
    # Wybierz losowo pewną liczbę krawędzi, aby stworzyć graf nieważony
    unweighted_edges = random.sample(possible_edges, num_edges)
    
    # Dodaj wybrane krawędzie do grafu
    G.add_edges_from(unweighted_edges)
    
    return G

# Funkcja do stworzenia grafu ważonego
def create_large_weighted_graph(num_nodes, num_edges):
    G = nx.Graph()
    G.add_nodes_from(range(num_nodes))
    
    # Tworzenie listy możliwych krawędzi (unikalnych par węzłów)
    possible_edges = [(i, j) for i in range(num_nodes) for j in range(i + 1, num_nodes)]
    
    # Wybierz losowo pewną liczbę krawędzi, aby stworzyć graf ważony
    weighted_edges = random.sample(possible_edges, num_edges)
    
    # Dodaj wybrane krawędzie do grafu z losowymi wagami
    for edge in weighted_edges:
        weight = random.randint(1, 10)  # Przypisanie losowej wagi od 1 do 10
        G.add_edge(edge[0], edge[1], weight=weight)

    plot_graph(G,"Large Weighted Graph")    
    
    return G


def create_weighted_graph(num_nodes, num_edges):
    G = nx.Graph()
    G.add_nodes_from(range(num_nodes))
    
    # Tworzenie listy możliwych krawędzi (unikalnych par węzłów)
    possible_edges = [(i, j) for i in range(num_nodes) for j in range(i + 1, num_nodes)]
    
    # Wybierz losowo pewną liczbę krawędzi, aby stworzyć graf ważony
    weighted_edges = random.sample(possible_edges, num_edges)
    
    # Dodaj wybrane krawędzie do grafu z losowymi wagami
    for edge in weighted_edges:
        weight = random.randint(1, 10)  # Przypisanie losowej wagi od 1 do 10
        G.add_edge(edge[0], edge[1], weight=weight)

    plot_graph(G,"Small Weighted Graph")    
    
    return G



def plot_graph(G, title):
    pos = nx.spring_layout(G, seed=42)  # Seed for consistent layout
    weights = nx.get_edge_attributes(G, 'weight')

    plt.figure(figsize=(20, 15))  # Increase figure size for better clarity
    nx.draw(
        G, pos, with_labels=True, node_color='lightblue', node_size=300, 
        font_size=8, edge_color='gray', width=0.5, alpha=0.7
    )
    nx.draw_networkx_edge_labels(G, pos, edge_labels=weights, font_size=6)  # Small font for edge labels

    plt.title(title)
    
    # Save as a high-resolution PNG
    filepath = f"{title}.png"
    plt.savefig(filepath, dpi=300, bbox_inches='tight')  # dpi=300 for high-resolution
    print(f"Graph saved as {filepath}")
    generated_files.append(filepath)

    plt.show()

# Funkcje do generowania różnych typów grafów
def create_large_sparse_graph(num_nodes, num_edges):
    G = nx.Graph()

    # Add nodes to the graph
    G.add_nodes_from(range(num_nodes))

    # Ensure the graph is connected by adding a spanning tree first
    nodes = list(range(num_nodes))
    random.shuffle(nodes)  # Shuffle nodes to create a random tree structure
    
    for i in range(num_nodes - 1):
        # Create a tree by connecting node i to i+1
        weight = random.randint(1, 50)
        G.add_edge(nodes[i], nodes[i + 1], weight=weight)

    # Now we have num_nodes - 1 edges, we can add more random edges to make it sparse
    possible_edges = [(i, j) for i in range(num_nodes) for j in range(i + 1, num_nodes) if not G.has_edge(i, j)]

    # Calculate how many more edges we can add
    additional_edges = num_edges - (num_nodes - 1)
    
    if additional_edges > 0:
        # Ensure we do not exceed the maximum number of edges possible for a simple graph
        max_additional_edges = len(possible_edges)
        if additional_edges > max_additional_edges:
            additional_edges = max_additional_edges
            
        sparse_edges = random.sample(possible_edges, additional_edges)
        for edge in sparse_edges:
            weight = random.randint(1, 50)
            G.add_edge(edge[0], edge[1], weight=weight)

    plot_graph(G, "Large Sparse Graph")

    return G

def create_small_sparse_graph(num_nodes, num_edges):
    G = nx.Graph()
    G.add_nodes_from(range(num_nodes))
    
    # Ensure connectivity by creating a spanning tree
    nodes = list(range(num_nodes))
    random.shuffle(nodes)
    
    for i in range(num_nodes - 1):
        weight = random.randint(1, 50)
        G.add_edge(nodes[i], nodes[i + 1], weight=weight)
    
    # Calculate additional edges needed to meet num_edges
    additional_edges = num_edges - (num_nodes - 1)
    
    if additional_edges > 0:
        possible_edges = [(i, j) for i in range(num_nodes) for j in range(i+1, num_nodes) if not G.has_edge(i, j)]
        sparse_edges = random.sample(possible_edges, additional_edges)
        
        for u, v in sparse_edges:
            weight = random.randint(1, 50)
            G.add_edge(u, v, weight=weight)
    
    plot_graph(G, "Small Sparse Graph")
    return G

def create_dense_graph(num_nodes):
    G = nx.Graph()
    
    # Dodaj węzły do grafu
    G.add_nodes_from(range(num_nodes))
    
    # Tworzenie krawędzi między większością par węzłów
    # W gęstym grafie liczba krawędzi zbliża się do maksymalnej liczby krawędzi
    for i in range(num_nodes):
        for j in range(i + 1, num_nodes):
            if random.random() > 0.2:  # Losowo dodaj krawędź z prawdopodobieństwem 0.8
                weight = random.randint(1, 10)  # Losowa waga w przedziale 1-10
                G.add_edge(i, j, weight=weight)

    plot_graph(G, "Large Dense Graph")            
    
    return G

def create_small_dense_graph(num_nodes):
    G = nx.Graph()
    
    # Dodaj węzły do grafu
    G.add_nodes_from(range(num_nodes))
    
    # Tworzenie krawędzi między większością par węzłów
    # W gęstym grafie liczba krawędzi zbliża się do maksymalnej liczby krawędzi
    for i in range(num_nodes):
        for j in range(i + 1, num_nodes):
            if random.random() > 0.2:  # Losowo dodaj krawędź z prawdopodobieństwem 0.8
                weight = random.randint(1, 10)  # Losowa waga w przedziale 1-10
                G.add_edge(i, j, weight=weight)

    plot_graph(G, "Small Dense Graph")            
    
    return G


def generate_large_weighted_graph(n, edges):
    G = nx.gnm_random_graph(n, edges)
    for (u, v) in G.edges():
        G.edges[u, v]['weight'] = random.randint(1, 100)
    return G


# Funkcja porównująca wybrane grafy i algorytmy
def compare_graphs(graphs, algorithms, start_node, goal_node):
    results = []
    for graph_name, graph in graphs.items():
        for alg_name, alg_func in algorithms.items():
            result, duration, memory_usage = measure_memory_and_time(alg_func, graph, start_node, goal_node)
            
            if result is None:
                result = []  # Handle None result safely

            path_length = len(result[0]) if isinstance(result[0], list) else 0

            results.append([graph_name, alg_name, duration, memory_usage, path_length, result[1]])
    
    return results



# Funkcja wybierająca węzły, które są w tej samej spójnej składowej
def get_connected_nodes(graph):
    components = list(nx.connected_components(graph))
    largest_component = max(components, key=len)  # Bierzemy największą spójną składową
    start_node = random.choice(list(largest_component))
    goal_node = random.choice(list(largest_component))
    
    # Debug: Sprawdzenie czy węzły są w tej samej spójnej składowej
    assert nx.has_path(graph, start_node, goal_node), f"No path between {start_node} and {goal_node}"
    
    return start_node, goal_node


# Funkcja do mierzenia pamięci i czasu działania algorytmu
def measure_memory_and_time(algorithm, graph, *args):
    tracemalloc.start()
    start_time = time.perf_counter()
    
    result = algorithm(graph, *args)

    print(result)
    
    end_time = time.perf_counter()
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    
    memory_usage = peak / 10**6  # Pamięć w MB
    return result, end_time - start_time, memory_usage