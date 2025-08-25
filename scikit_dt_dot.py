import numpy as np
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn.model_selection import train_test_split
from sklearn import metrics
import re
import os
import pickle  # For saving and loading models
import graphviz 

def parse_input_file(filename):
    """Parse input vectors from text files."""
    inputs = []
    with open(filename, 'r') as file:
        content = file.read()
    # This pattern to find vectors of 5 floating-point numbers
    pattern = r'\[([-+]?\d*\.?\d+),\s*([-+]?\d*\.?\d+),\s*([-+]?\d*\.?\d+),\s*([-+]?\d*\.?\d+),\s*([-+]?\d*\.?\d+)\]'
    matches = re.findall(pattern, content)
    for match in matches:
        # Convert matched strings to float and append to the list
        input_values = [float(val) for val in match]
        inputs.append(input_values)
    return inputs

def load_data_from_file(filename):
    """Load data from a single specified text file."""
    if not os.path.exists(filename):
        print(f"File not found: {filename}")
        return np.array([])
    print(f"Processing file: {filename}")
    inputs = parse_input_file(filename)
    print(f"Found {len(inputs)} inputs.")
    return np.array(inputs) if inputs else np.array([])

adversarial_filename = "prop_8_data/adv_merged.txt"
non_adversarial_filename = "prop_8_data/nonadv_merged.txt"

# print(f"Loading adversarial data from '{adversarial_filename}'")
adversarial_data = load_data_from_file(adversarial_filename)

# print(f"\nLoading non-adversarial data from '{non_adversarial_filename}'")
non_adversarial_data = load_data_from_file(non_adversarial_filename)

if len(adversarial_data) == 0 or len(non_adversarial_data) == 0:
    print("\nError: Both data files must contain data to proceed.")
    exit()

# print(f"\nSuccessfully loaded {len(adversarial_data)} adversarial and {len(non_adversarial_data)} non-adversarial inputs.")
# Note: Duplicates are not being removed as per your last version.
# To re-enable, uncomment the following two lines:
# adversarial_data = np.unique(adversarial_data, axis=0)
# non_adversarial_data = np.unique(non_adversarial_data, axis=0)
# print(f" -> After removing duplicates: {len(adversarial_data)} adversarial and {len(non_adversarial_data)} non-adversarial inputs remain.")

#Combine Data for Training
X = np.vstack((adversarial_data, non_adversarial_data))
y = np.array([1] * len(adversarial_data) + [0] * len(non_adversarial_data))
print(f"\nTotal samples for training: {len(X)}")

# Train or Load the Decision Tree Classifier
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)
print("Data split into training (75%) and testing (25%) sets.")

model_filename = "decision_tree_model.pkl"
force_retrain = False  # Set to True to retrain even if a model file exists

if os.path.exists(model_filename) and not force_retrain:
    print(f"Loading existing trained model from '{model_filename}'")
    with open(model_filename, 'rb') as file:
        clf = pickle.load(file)
    print("Model loaded successfully")
else:
    if force_retrain:
        print("Force retraining enabled. Training a new model")
    else:
        print("No existing model found. Training a new model")
    
    clf = DecisionTreeClassifier(criterion="entropy", max_depth=6, random_state=42, class_weight='balanced')
    
    print("Training the Decision Tree")
    clf.fit(X_train, y_train)
    print("Model training complete")
    
    print(f"Saving trained model to '{model_filename}'")
    with open(model_filename, 'wb') as file:
        pickle.dump(clf, file)
    print("Model saved successfully")

#Evaluate the Model
y_pred = clf.predict(X_test)
accuracy = metrics.accuracy_score(y_test, y_pred)   #uses accuracy_score from sklearn.metrics

print(f"Model Accuracy on Test Data: {accuracy:.2%}")


# print("\nGenerating tree with Graphviz")
feature_names = ['ρ', 'θ', 'ψ', 'v_own', 'v_int']

dot_data = export_graphviz(
    clf,
    out_file=None,
    feature_names=feature_names,
    class_names=['Non-Adversarial (UNSAT)', 'Adversarial (SAT)'],
    filled=True,
    rounded=True,
    special_characters=True,
    proportion=True,
    impurity=True
)

#create graph from the DOT data
graph = graphviz.Source(dot_data)

#output decision tree to a file dt_plot_dot
graph.render("prop_8_data/dt_plot_dot", format='png', view=False, cleanup=True)

#Classifying a new input
new_input = np.array([-0.016045, -0.489653, -0.009506, 0.243222, 0.274097]).reshape(1, -1)
prediction = clf.predict(new_input)

if prediction[0] == 1:
    class_label = "Adversarial (SAT)"
else:
    class_label = "Non-Adversarial (UNSAT)"

print(f"The new input {new_input.flatten()} is classified as: {class_label}")
