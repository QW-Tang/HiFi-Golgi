import pandas as pd
import numpy as np

# Read SWC file
def load_swc(file_path):
    return pd.read_csv(file_path, sep=' ', header=None, names=['n', 'type', 'x', 'y', 'z', 'r', 'parent'], comment='#')


# Recursive function: calculate branch length
def calculate_branch_length(index):
    current_node = nodes[index]
    children_indices = data[data['parent'] == index + 1].index  # Find all child nodes
    total_length = 0
    for child_index in children_indices:
        child_node = nodes[child_index]
        branch_length = np.linalg.norm(current_node - child_node)
        total_length += branch_length + calculate_branch_length(child_index)
    return total_length


# Recursive function: count number of branches
def count_branches(index):
    children_indices = data[data['parent'] == index + 1].index  # Find all child nodes
    if len(children_indices) == 0:  # Leaf node (no children)
        return 1
    count = 0
    for child_index in children_indices:
        count += count_branches(child_index)
    return count


# Main program
if __name__ == "__main__":
    file_path = ('  ')
    data = load_swc(file_path)
    data.reset_index(drop=True, inplace=True)

    # Find root node (parent == -1)
    root_nodes = data[data['parent'] == -1]
    if root_nodes.empty:
        raise ValueError("Root node not found. Please check if the SWC file format is correct!")
    root_index = root_nodes.index[0]

    # Extract node coordinates, radii, and parent information
    nodes = data[['x', 'y', 'z']].values
    radii = data['r'].values
    parents = data['parent'].values

    # Calculate total branch length
    total_length = calculate_branch_length(root_index)

    # Count number of branches
    branch_count = count_branches(root_index)

    # Output results
    print(f"Number of branches: {branch_count}")
    print(f"Total neuronal branch length: {total_length:.2f}")
