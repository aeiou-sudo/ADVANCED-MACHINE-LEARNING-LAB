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
    if len(class_labels) == 1:
        return class_labels[0]
    if not attributes:
        return data.iloc[:, -1].mode()[0]
    best_attr = best_attribute(data, attributes)
    tree = {best_attr: {}}
    remaining_attributes = [attr for attr in attributes if attr != best_attr]
    for value in data[best_attr].unique():
        subset = data[data[best_attr] == value]
        if subset.empty:
            tree[best_attr][value] = data.iloc[:, -1].mode()[0]
        else:
            tree[best_attr][value] = build_tree(subset, remaining_attributes)
    return tree

# Classify a new sample
def classify(tree, sample):
    if not isinstance(tree, dict):
        return tree
    root = next(iter(tree))
    value = sample[root]
    if value not in tree[root]:
        return "Unknown"  # Handle unseen values gracefully
    return classify(tree[root][value], sample)

# Simplified tree visualization
def plot_tree_simple(tree, dot=None, parent=None, label=None):
    if dot is None:
        dot = Digraph(format='png', graph_attr={'rankdir': 'TB'})
    if isinstance(tree, dict):
        root = next(iter(tree))
        node_id = str(id(root))
        dot.node(node_id, root, shape='ellipse', style='filled', color='lightblue')
        if parent:
            dot.edge(parent, node_id, label=label)
        for value, subtree in tree[root].items():
            plot_tree_simple(subtree, dot, parent=node_id, label=str(value))
    else:
        leaf_id = str(id(tree))
        dot.node(leaf_id, str(tree), shape='box', style='filled', color='lightgreen' if tree == 'yes' else 'lightpink')
        dot.edge(parent, leaf_id, label=str(label))
    return dot

# Main script
file_path = 'data.csv'
data = pd.read_csv(file_path)

# Normalize the dataset (case-insensitive)
data = data.applymap(lambda x: x.lower() if isinstance(x, str) else x)

# Prepare attributes
attributes = list(data.columns[:-1])

# Build decision tree
tree = build_tree(data, attributes)
print("Decision tree built successfully!")

# Visualize the tree
dot = plot_tree_simple(tree)
dot.render("decision_tree", format="png", cleanup=True)
dot.view()

# Classify a new sample
print("\nProvide input values for classification:")
new_sample = {}
for attr in attributes:
    new_sample[attr] = input(f"Enter value for {attr}: ").strip().lower()

# Predict class
predicted_class = classify(tree, new_sample)
print(f"\nPredicted Class: {predicted_class}")
