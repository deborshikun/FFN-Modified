import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn import metrics
import matplotlib.pyplot as plt
from sklearn.tree import plot_tree
import re
import glob
import os
import pickle  # For saving and loading models

def parse_input_file(filename):
    """Parse input vectors from text files."""
    inputs = []
    with open(filename, 'r') as file:
        content = file.read()
        
    # This pattern to find vectors of 5 floating-point numbers
    pattern = r'\[([-+]?\d*\.?\d+),\s*([-+]?\d*\.?\d+),\s*([-+]?\d*\.?\d+),\s*([-+]?\d*\.?\d+),\s*([-+]?\d*\.?\d+)\]'
    matches = re.findall(pattern, content)
    # print(matches)    
    
    for match in matches:
        # Convert matched strings to float and append to the list
        input_values = [float(val) for val in match]
        inputs.append(input_values)
             
    return inputs

def load_data_from_file(filename):
    """Load data from a single specified text file."""
    all_inputs = []
    if not os.path.exists(filename):
        print(f"File not found: {filename}")
        return np.array([])
        
    print(f"Processing file: {filename}")
    inputs = parse_input_file(filename)
    all_inputs.extend(inputs)
    print(f"  -> Found {len(inputs)} inputs.")
    
    return np.array(all_inputs) if all_inputs else np.array([])


# Define the filenames
adversarial_filename = "adv_merged.txt"
non_adversarial_filename = "nonadv_merged.txt"

# Load adversarial data (SAT cases) = Class 1
print(f"Loading adversarial data from '{adversarial_filename}'...")
adversarial_data = load_data_from_file(adversarial_filename)

# Load non-adversarial data (UNSAT cases) = Class 0
print(f"\nLoading non-adversarial data from '{non_adversarial_filename}'...")
non_adversarial_data = load_data_from_file(non_adversarial_filename)

if len(adversarial_data) > 0:
    print(f"\nSuccessfully loaded {len(adversarial_data)} adversarial inputs.")
    # adversarial_data = np.unique(adversarial_data, axis=0) #removing dupes
    # print(f" -> After removing duplicates: {len(adversarial_data)} unique adversarial inputs remain.")
else:
    print("\nWarning: No adversarial data was loaded. The model cannot be trained without it.")
    exit()

if len(non_adversarial_data) > 0:
    print(f"\nSuccessfully loaded {len(non_adversarial_data)} non-adversarial inputs.")
    # non_adversarial_data = np.unique(non_adversarial_data, axis=0) #removing dupes
    # print(f" -> After removing duplicates: {len(non_adversarial_data)} unique non-adversarial inputs remain.")
else:
    print("\nWarning: No non-adversarial data was loaded. The model cannot be trained without it.")
    exit()

print(f"\nTotal adversarial samples for training: {len(adversarial_data)}")
print(f"Total non-adversarial samples for training: {len(non_adversarial_data)}")


# Combine the datasets into a single feature matrix (X)
X = np.vstack((adversarial_data, non_adversarial_data))

# Creating the corresponding labels (y): 1 for adversarial, 0 for non-adversarial
y = np.array([1] * len(adversarial_data) + [0] * len(non_adversarial_data))

print(f"Total samples: {len(X)}")
print(f"Feature count per sample: {X.shape[1]}")
print(f"Class distribution - Adversarial (1): {np.sum(y)}, Non-Adversarial (0): {len(y) - np.sum(y)}")
print(f"Unique labels in y: {np.unique(y)}")
print(f"Label counts: {np.bincount(y)}")

# Check first few samples of each class
# print(f"\nFirst 3 adversarial samples:\n{adversarial_data[:3] if len(adversarial_data) > 0 else 'None'}")
# print(f"\nFirst 3 non-adversarial samples:\n{non_adversarial_data[:3] if len(non_adversarial_data) > 0 else 'None'}")


# test_size=0.25 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)

print("\nData split into training (75%) and testing (25%) sets")

# Create a Decision Tree Classifier object
model_filename = "decision_tree_model.pkl"
force_retrain = True  # True if you want to retrain even if model exists

# Check if a trained model already exists
if os.path.exists(model_filename) and not force_retrain:
    print(f"Loading existing trained model from '{model_filename}'...")
    with open(model_filename, 'rb') as file:
        clf = pickle.load(file)
    print("Model loaded successfully!")
    
else:
    if force_retrain:
        print("Force retraining enabled. Training a new model...")
    else:
        print("No existing model found. Training a new model...")
    
    clf = DecisionTreeClassifier(criterion="entropy", max_depth=6, random_state=42, class_weight='balanced') #class_weight='balanced' to handle class imbalance
    #criterion='entropy' to use Information Gain for splits

    # Train the classifier on the training data
    print("Training the Decision Tree model")
    clf.fit(X_train, y_train)
    print("Model training complete.")
    
    # Save the trained model
    print(f"Saving trained model to '{model_filename}'...")
    with open(model_filename, 'wb') as file:
        pickle.dump(clf, file)
    print("Model saved successfully!")
    print("Training the Decision Tree model")
    clf.fit(X_train, y_train)
    print("Model training complete.")
    
    # Save the trained model
    print(f"Saving trained model to '{model_filename}'...")
    with open(model_filename, 'wb') as file:
        pickle.dump(clf, file)
    print("Model saved successfully!")

# Predicting the class labels for the test dataset
y_pred = clf.predict(X_test)

# Calculate and print the model's accuracy
accuracy = metrics.accuracy_score(y_test, y_pred)   #uses accuracy_score from sklearn.metrics
print(f"Model Accuracy on Test Data: {accuracy:.4%}")

# visualisation of the decision tree
plt.figure(figsize=(80, 50))  
plot_tree(
    clf, 
    filled=True, 
    feature_names=['ρ', 'θ', 'ψ', 'v_own', 'v_int'], 
    class_names=['Non-Adversarial (UNSAT)', 'Adversarial (SAT)'], 
    rounded=True,
    fontsize=13,  
    impurity=True,
    proportion=True,
    precision=3  # 3 decimal places for numbers
)
plt.title("Trained Decision Tree for Adversarial Input Classification", fontsize=48)
plt.tight_layout()  # Better spacing
plt.savefig("dt_plot.png", dpi=200, bbox_inches='tight', facecolor='white')
plt.show()
# Classifying new input sample
new_input = np.array([-0.016045, -0.489653, -0.009506, 0.243222, 0.274097]).reshape(1, -1)
prediction = clf.predict(new_input)

if prediction == 1:
    class_label = "Adversarial (SAT)"
else:
    class_label = "Non-Adversarial (UNSAT)"

print(f"The new input {new_input.flatten()} is classified as: {class_label}")
