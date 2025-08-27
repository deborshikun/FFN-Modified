"""
Extract decision tree split conditions using a depth-first search (DFS) traversal
and write them to a text file, one per line, in traversal order.
"""

import pickle
from pathlib import Path

model_path = Path("decision_tree_model.pkl")
output_path = Path("tree_conditions_paths.txt")

def load_model(model_path):
    with model_path.open('rb') as f:
        return pickle.load(f)

def get_feature_names(tree_estimator):
    if hasattr(tree_estimator, 'feature_names_in_'):
        return [str(x) for x in list(tree_estimator.feature_names_in_)]
    if hasattr(tree_estimator, 'n_features_in_'):
        n = int(tree_estimator.n_features_in_)
        return [f'feature_{i}' for i in range(n)]
    tree = tree_estimator.tree_
    feat = getattr(tree, 'feature', None)
    if feat is not None:
        import numpy as np
        valid = feat[feat >= 0]
        n = int(np.max(valid)) + 1 if valid.size > 0 else 0
        return [f'feature_{i}' for i in range(n)]
    return []

def extract_paths(tree_estimator):
    tree = tree_estimator.tree_
    feature = tree.feature
    threshold = tree.threshold
    children_left = tree.children_left
    children_right = tree.children_right
    feature_names = get_feature_names(tree_estimator)
    paths = []

    def dfs(node_id, path):
        left = children_left[node_id]
        right = children_right[node_id]
        if left == -1 and right == -1:
            # Leaf node: save the path
            paths.append(', '.join(path))
            return
        feat_idx = int(feature[node_id])
        feat_name = (
            feature_names[feat_idx]
            if feat_idx < len(feature_names)
            else f'feature_{feat_idx}'
        )
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
