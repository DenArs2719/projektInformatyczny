import networkx as nx
import tkinter as tk
from tkinter import ttk
import pandas as pd
from graphs import graphs
from algorithm import algorithm
import matplotlib.pyplot as plt
import os
import atexit

def plot_results(data):
    df = pd.DataFrame(data, columns=["Graph", "Algorithm", "Execution Time (s)"])
    pivot_df = df.pivot(index="Graph", columns="Algorithm", values="Execution Time (s)")
    pivot_df.plot(kind="bar", figsize=(10, 7))
    plt.ylabel("Execution Time (s)")
    plt.title("Comparison of Algorithms on Different Graphs")
    plt.show()

def cleanup_generated_files():
    for filepath in graphs.generated_files:
        if os.path.exists(filepath):
            os.remove(filepath)
            print(f"Removed file: {filepath}")

# Function to display results in a window
def show_results_in_window(data, title):
    df = pd.DataFrame(data, columns=["Graph", "Algorithm", "Execution Time (s)", "Memory Usage (MB)", "Path Length", "Total Weight"])

    # Create a new window for the results
    window = tk.Toplevel(root)
    window.title(title)

    table_frame = ttk.Frame(window)
    table_frame.pack(padx=10, pady=10)

    # Updated table with additional columns
    table = ttk.Treeview(table_frame, columns=("Graph", "Algorithm", "Execution Time", "Memory Usage", "Path Length", "Total Weight"), show="headings")
    table.heading("Graph", text="Graph")
    table.heading("Algorithm", text="Algorithm")
    table.heading("Execution Time", text="Execution Time (s)")
    table.heading("Memory Usage", text="Memory Usage (MB)")
    table.heading("Path Length", text="Path Length") 
    table.heading("Total Weight", text="Total Weight")

    # Add data to the table
    for row in data:
        table.insert("", "end", values=row)

    table.pack()

    # Create a plot for the results
    pivot_df = df.pivot(index="Graph", columns="Algorithm", values="Execution Time (s)")
    pivot_df.plot(kind="bar", figsize=(10, 7))
    plt.ylabel("Execution Time (s)")
    plt.title(title)
    plt.show()

# Function to run the comparison
def run_comparison():
    selected_graphs = [graph_listbox.get(i) for i in graph_listbox.curselection()]
    if len(selected_graphs) > 4:
        tk.messagebox.showerror("Error", "You can select up to 4 graphs.")
        return

    filtered_graphs = {name: all_graphs[name] for name in selected_graphs}

    algorithms = {
        "Dijkstra": algorithm.dijkstra,
        "BFS": algorithm.bfs,
        "DFS": algorithm.dfs,
        "A*": algorithm.a_star
    }

    # Get user-provided start and goal nodes
    try:
        start_node = int(start_node_entry.get())
        goal_node = int(goal_node_entry.get())
    except ValueError:
        tk.messagebox.showerror("Error", "Start and Goal nodes must be integers.")
        return

    # Check if start and goal nodes are valid for each selected graph
    for graph_name, graph in filtered_graphs.items():
        if start_node not in graph.nodes() or goal_node not in graph.nodes():
            tk.messagebox.showerror("Error", f"Start or Goal node is invalid for {graph_name}")
            return

    # Perform the comparison using the provided start and goal nodes
    results = graphs.compare_graphs(filtered_graphs, algorithms, start_node, goal_node)
    show_results_in_window(results, "Performance and Memory Comparison")
    plot_results(results)

# Create the UI
root = tk.Tk()
root.title("Graph Comparison Tool")

all_graphs = graphs.generate_graphs()

# List of graphs to choose from
graph_listbox = tk.Listbox(root, selectmode=tk.MULTIPLE)
for graph_name in all_graphs.keys():
    graph_listbox.insert(tk.END, graph_name)
graph_listbox.pack()

# Input fields for start and goal nodes
start_node_label = tk.Label(root, text="Start Node:")
start_node_label.pack()
start_node_entry = tk.Entry(root)
start_node_entry.pack()

goal_node_label = tk.Label(root, text="Goal Node:")
goal_node_label.pack()
goal_node_entry = tk.Entry(root)
goal_node_entry.pack()

# Button to run the comparison
compare_button = tk.Button(root, text="Compare Selected Graphs", command=run_comparison)
compare_button.pack(pady=10)

root.mainloop()

# Register the cleanup function to be called at application exit
atexit.register(cleanup_generated_files)
