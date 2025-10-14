"""
Extract decision tree split conditions using a depth-first search (DFS) traversal
and write them to a text file, one per line, in traversal order, including the
resultant class at each leaf.
"""

import pickle
from pathlib import Path
import numpy as np

# Resolve paths relative to this script's directory for robustness
BASE_DIR = Path(__file__).parent
model_path = BASE_DIR / "decision_tree_model.pkl"
output_path = BASE_DIR / "tree_conditions_paths.txt"

def load_model(model_path):
    with model_path.open('rb') as f:
        return pickle.load(f)

def extract_paths(tree_estimator):
    tree = tree_estimator.tree_
    feature = tree.feature
    threshold = tree.threshold
    children_left = tree.children_left
    children_right = tree.children_right

    # Map class indices to readable labels
    label_map = {0: 'Non-Adversarial (UNSAT)', 1: 'Adversarial (SAT)'}
    classes = getattr(tree_estimator, 'classes_', None)

    def leaf_label(node_id):
        # tree.value has shape (n_nodes, n_outputs, n_classes)
        values = tree.value[node_id][0]
        pred_idx = int(np.argmax(values))
        if classes is not None:
            pred_class = classes[pred_idx]
        else:
            pred_class = pred_idx
        # Ensure integer keys for label_map
        try:
            return label_map[int(pred_class)]
        except Exception:
            return str(pred_class)

    paths = []

    def dfs(node_id, path):
        left = children_left[node_id]
        right = children_right[node_id]
        if left == -1 and right == -1:
            # Leaf node: save the path with resultant class label
            paths.append(', '.join(path) + f", class = {leaf_label(node_id)}")
            return
        feat_idx = int(feature[node_id])
        feat_name = f"X_{feat_idx}"
        thr = float(threshold[node_id])
        # Left child: condition is <=
        dfs(left, path + [f"{feat_name} <= {thr:.6g}"])
        # Right child: condition is >
        dfs(right, path + [f"{feat_name} > {thr:.6g}"])

    dfs(0, [])
    return paths


tree_estimator = load_model(model_path)
paths = extract_paths(tree_estimator)
output_path.parent.mkdir(parents=True, exist_ok=True)
with output_path.open('w', encoding='utf-8') as f:
    for i, path in enumerate(paths, 1):
        f.write(f"{i}. {path}\n")
print(f"Wrote {len(paths)} paths to: {output_path}")
