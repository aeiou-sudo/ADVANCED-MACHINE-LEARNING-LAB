import pandas as pd
import math
from graphviz import Digraph

# Calculate entropy
def entropy(data):
    total = len(data)
    class_counts = data.iloc[:, -1].value_counts()
    ent = 0
    for count in class_counts:
        p = count / total
        ent -= p * math.log2(p)
    return ent

# Calculate information gain
def information_gain(data, attribute):
    total_entropy = entropy(data)
    total = len(data)
    values = data[attribute].unique()
    weighted_entropy = 0
    for value in values:
        subset = data[data[attribute] == value]
        weighted_entropy += (len(subset) / total) * entropy(subset)
    return total_entropy - weighted_entropy

# Find the best attribute
def best_attribute(data, attributes):
    gains = {attr: information_gain(data, attr) for attr in attributes}
    return max(gains, key=gains.get)

# Build the decision tree
def build_tree(data, attributes):
    class_labels = data.iloc[:, -1].unique()
    print(f"Building tree... Class labels: {class_labels}")
    if len(class_labels) == 1:
        print(f"Only one class label: {class_labels[0]}")
        return class_labels[0]
    if not attributes:
        mode_class = data.iloc[:, -1].mode()[0]
        print(f"No attributes left. Returning most common class: {mode_class}")
        return mode_class
    best_attr = best_attribute(data, attributes)
    print(f"Best attribute: {best_attr}")
    tree = {best_attr: {}}
    remaining_attributes = [attr for attr in attributes if attr != best_attr]
    for value in data[best_attr].unique():
        print(f"Processing value '{value.strip()}' of attribute '{best_attr}'")
        subset = data[data[best_attr] == value]
        if subset.empty:
            mode_class = data.iloc[:, -1].mode()[0]
            print(f"Subset is empty for '{value}'. Returning most common class: {mode_class}")
            tree[best_attr][value.strip()] = mode_class
        else:
            tree[best_attr][value.strip()] = build_tree(subset, remaining_attributes)
    return tree

# Classify a new sample
def classify(tree, sample, data):
    print(f"Classifying sample: {sample}")
    if not isinstance(tree, dict):
        print(f"Reached leaf node: {tree}")
        return tree
    root = next(iter(tree))
    value = sample[root]
    print(f"Root: {root}, Sample value: {value}")
    
    # Check if the value exists in the tree path
    if value.strip() not in tree[root]:  # Strip spaces from both the tree and input
        print(f"Value '{value.strip()}' not found in tree. Returning most common class.")
        return data.iloc[:, -1].mode()[0]  # Fallback to the most common class
    
    return classify(tree[root][value.strip()], sample, data)

def plot_tree_simple(tree, dot=None, parent=None, label=None, node_counter=None):
    if node_counter is None:
        node_counter = [0]  # List to maintain a counter between recursive calls
    
    if dot is None:
        dot = Digraph(format='png', graph_attr={'rankdir': 'TB'})
    
    if isinstance(tree, dict):  # Non-leaf node
        root = next(iter(tree))  # Get the root attribute of this subtree
        node_id = f"{root}_{node_counter[0]}"  # Unique ID based on the root attribute and a counter
        dot.node(node_id, root, shape='ellipse', style='filled', color='lightblue')
        
        if parent:
            dot.edge(parent, node_id, label=label)
        
        node_counter[0] += 1  # Increment the node counter for the next node
        
        for value, subtree in tree[root].items():
            plot_tree_simple(subtree, dot, parent=node_id, label=str(value), node_counter=node_counter)
    else:  # Leaf node
        leaf_id = f"leaf_{node_counter[0]}"  # Unique leaf ID based on the counter
        dot.node(leaf_id, str(tree), shape='box', style='filled', color='lightgreen' if tree == 'yes' else 'lightpink')
        
        if parent:
            dot.edge(parent, leaf_id, label=str(label))
        
        node_counter[0] += 1  # Increment the node counter
    
    return dot


# Main script
file_path = 'data.csv'
data = pd.read_csv(file_path)

# Strip any extra spaces in the column names
data.columns = data.columns.str.strip()

# Normalize the dataset (case-insensitive)
data = data.applymap(lambda x: x.lower() if isinstance(x, str) else x)

# Prepare attributes
attributes = list(data.columns[:-1])

# Build decision tree
print("\nBuilding decision tree...")
tree = build_tree(data, attributes)
print("\nDecision tree built successfully!")

# Visualize the tree
dot = plot_tree_simple(tree)
dot.render("decision_tree", format="png", cleanup=True)
dot.view()

#Main MENU
while True:
    # Classify a new sample
    print("\nProvide input values for classification:")
    new_sample = {}
    for attr in attributes:
        new_sample[attr] = input(f"Enter value for {attr}: ").strip().lower()  # Strip extra spaces and convert to lower

    # Predict class
    predicted_class = classify(tree, new_sample, data)
    print(f"\nPredicted Class: {predicted_class}")
