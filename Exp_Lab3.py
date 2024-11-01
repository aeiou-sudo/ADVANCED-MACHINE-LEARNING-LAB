import pandas as pd
import numpy as np
from collections import Counter
import math

# Function to calculate entropy
def entropy(target_col):
    elements, counts = np.unique(target_col, return_counts=True)
    entropy_value = sum(
        [(-counts[i]/sum(counts)) * math.log2(counts[i]/sum(counts)) for i in range(len(elements))]
    )
    return entropy_value

# Function to calculate Information Gain
def information_gain(data, split_attribute_name, target_name="Class"):
    total_entropy = entropy(data[target_name])
    vals, counts = np.unique(data[split_attribute_name], return_counts=True)
    
    weighted_entropy = sum(
        [(counts[i]/sum(counts)) * entropy(data.where(data[split_attribute_name]==vals[i]).dropna()[target_name]) 
         for i in range(len(vals))]
    )
    
    information_gain_value = total_entropy - weighted_entropy
    return information_gain_value

# Function to build the tree using the ID3 algorithm
def id3(data, original_data, features, target_attribute_name="class", parent_node_class=None):
    # Check if all target values have the same value
    if len(np.unique(data[target_attribute_name])) <= 1:
        return np.unique(data[target_attribute_name])[0]
    
    # If dataset is empty, return the parent node class (majority class of original dataset)
    elif len(data) == 0:
        # Use the majority class from the original dataset if subset is empty
        return np.unique(original_data[target_attribute_name])[np.argmax(
            np.unique(original_data[target_attribute_name], return_counts=True)[1])]
    
    # If no more features to split, return majority class of current data
    elif len(features) == 0:
        return parent_node_class
    
    # Recursive case
    else:
        # Majority class of current node to use as default in case of empty subset
        parent_node_class = np.unique(data[target_attribute_name])[np.argmax(
            np.unique(data[target_attribute_name], return_counts=True)[1])]
        
        # Select feature with the highest Information Gain
        item_values = [information_gain(data, feature, target_attribute_name) for feature in features]
        best_feature_index = np.argmax(item_values)
        best_feature = features[best_feature_index]
        
        # Create tree structure
        tree = {best_feature: {}}
        
        # Remove selected feature from list of features
        features = [i for i in features if i != best_feature]
        
        # Grow tree branches based on best feature values
        for value in np.unique(data[best_feature]):
            # Create subset for this branch
            sub_data = data.where(data[best_feature] == value).dropna()
            
            # If subset is empty, add a leaf with majority class
            if sub_data.empty:
                tree[best_feature][value] = parent_node_class
            else:
                # Recursive call for non-empty subset
                subtree = id3(sub_data, original_data, features, target_attribute_name, parent_node_class)
                tree[best_feature][value] = subtree
        
        return tree


# Function to classify a new sample using the generated tree
def classify(sample, tree):
    for attribute in tree.keys():
        value = sample[attribute]
        tree = tree[attribute][value]
        if type(tree) is not dict:
            return tree

# Load the dataset from a file
file_path = './Dataset/weather_data.csv'  # Replace with the path to your dataset file
data = pd.read_csv(file_path)

# Features and target
features = list(data.columns[:-1])  # All columns except the target
target = 'Class'  # Modify based on the target column name in the file

# Build decision tree
tree = id3(data, data, features, target)

# Display the tree
print("Generated Decision Tree:")
print(tree)

# Classify a new sample
new_sample = {'Outlook': 'Rain', 'Temperature': 'Cool', 'Humidity': 'High', 'Wind': 'Strong'}  # Modify based on dataset attributes
classification = classify(new_sample, tree)

print("\nClassification of the new sample:", classification)
