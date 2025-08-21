import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier, plot_tree

def load_data(filepath):
    """Loads data from a text file, parsing each line as a list of floats."""
    data = []
    with open(filepath, 'r') as f:
        for line in f:
            try:
                # Strip brackets, split by comma, and convert to float
                processed_line = [float(x.strip()) for x in line.strip()[1:-1].split(',')]
                if len(processed_line) == 5:
                    data.append(processed_line)
            except (ValueError, IndexError) as e:
                print(f"Skipping malformed line: {line.strip()} due to error: {e}")
    return np.array(data)

# Define feature and class names for clear visualization
feature_names = ['ρ', 'θ', 'ψ', 'v_own', 'v_int']
class_names = ['UNSAT', 'SAT'] # UNSAT = non_adversarial, SAT = adversarial

# Load adversarial (SAT) and non-adversarial (UNSAT) data
try:
    X_adv = load_data('adv_merged.txt')
    X_non_adv = load_data('non_adv_merged.txt')

    # Create labels: 0 for non-adversarial (UNSAT), 1 for adversarial (SAT)
    y_adv = np.ones(X_adv.shape[0])
    y_non_adv = np.zeros(X_non_adv.shape[0])

    # Combine datasets
    X = np.vstack((X_adv, X_non_adv))
    y = np.concatenate((y_adv, y_non_adv))

    print(f"Loaded {X_adv.shape[0]} adversarial samples.")
    print(f"Loaded {X_non_adv.shape[0]} non-adversarial samples.")
    print(f"Total samples: {X.shape[0]}")


    # Initialize the classifier with a max depth of 6
    # random_state is used for reproducibility
    clf = DecisionTreeClassifier(max_depth=6, random_state=42)

    # Train the model
    clf.fit(X, y)

    print("\nGenerating decision tree plot...")
    plt.figure(figsize=(25, 15)) # Use a large figure size for readability

    plot_tree(clf,
              filled=True,
              feature_names=feature_names,
              class_names=class_names,
              rounded=True,
              fontsize=10,
              precision=3) # Use 3 decimal places for thresholds

    # Save the plot to a high-resolution PNG file
    output_filename = 'dt_plot_gemini.png'
    plt.savefig(output_filename, dpi=300, bbox_inches='tight')
    plt.close() # Close the plot to free memory

    print(f"Decision tree saved as '{output_filename}'")

except FileNotFoundError as e:
    print(f"Error: {e}. Please make sure 'adv_merged.txt' and 'non_adv_merged.txt' are in the same directory.")
except Exception as e:
    print(f"An unexpected error occurred: {e}")