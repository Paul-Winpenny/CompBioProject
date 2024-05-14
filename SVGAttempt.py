import time
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV, train_test_split

# Function to parse data from a file
def parse_data(file_path):
    sequences = []
    labels = []
    with open(file_path, "r") as file:
        for line in file:
            cleaned_line = line.strip()
            if cleaned_line in {"end", "<end>"}:
                sequences.append("end")
                labels.append("end")
            elif len(cleaned_line.split()) == 2:
                parts = cleaned_line.split()
                sequences.append(parts[0])  # Amino acid
                labels.append(parts[1])  # Secondary structure label
            else:
                continue  # Skip any malformed lines
    return sequences, labels

# Function to encode sequences
def encode_sequences(sequences):
    amino_acid_mapping = {'A': 0, 'R': 1, 'N': 2, 'D': 3, 'C': 4, 'Q': 5, 'E': 6, 'G': 7, 'H': 8, 'I': 9, 'L': 10, 'K': 11, 'M': 12, 'F': 13, 'P': 14, 'S': 15, 'T': 16, 'W': 17, 'Y': 18, 'V': 19, 'end': 20}
    encoded_seqs = np.zeros((len(sequences), 21))
    for idx, amino_acid in enumerate(sequences):
        encoded_seqs[idx, amino_acid_mapping[amino_acid]] = 1
    return encoded_seqs

# Function to encode labels
def encode_labels(labels):
    label_mapping = {'e': 0, 'h': 1, '_': 2, 'end': 3}
    encoded_labels = np.zeros(len(labels))  # Encode labels as integers
    for idx, label in enumerate(labels):
        if label != 'end':
            encoded_labels[idx] = label_mapping[label]
    return encoded_labels

print("Loading and processing data...")
start_time = time.time()
# Load and process training and test data
seq_train, labels_train = parse_data("E:\BioProjCopy\CompBioProject\protein-secondary-structure.train.txt")
seq_test, labels_test = parse_data("E:\BioProjCopy\CompBioProject\protein-secondary-structure.test.txt")

eseq_train = encode_sequences(seq_train)
elab_train = encode_labels(labels_train)
eseq_test = encode_sequences(seq_test)
elab_test = encode_labels(labels_test)

# Function to create sliding windows
def create_sliding_windows(encoded_seqs, window_size=13):
    padding = np.zeros((window_size // 2, 21))
    padded_seqs = np.vstack([padding, encoded_seqs, padding])
    windows = np.array([padded_seqs[i:i + window_size].flatten() for i in range(len(encoded_seqs))])
    return windows

print("Preparing windowed input for model...")
# Prepare windowed input for model
windowed_input_train = create_sliding_windows(np.array(eseq_train))
windowed_input_test = create_sliding_windows(np.array(eseq_test))

X_train = windowed_input_train
y_train = elab_train
X_test = windowed_input_test
y_test = elab_test

# Hyperparameter tuning using GridSearchCV
print("Defining and tuning the SVM model...")
param_grid = {
    'C': [0.1, 1, 10],
    'gamma': ['scale', 'auto'],
    'kernel': ['linear', 'rbf']
}

grid_search = GridSearchCV(SVC(probability=True), param_grid, cv=5, verbose=1, n_jobs=-1)
fit_start_time = time.time()
grid_search.fit(X_train, y_train)
fit_end_time = time.time()
print(f"Grid search complete. Time taken: {fit_end_time - fit_start_time} seconds.")

# Best parameters and estimator
best_params = grid_search.best_params_
best_svm_model = grid_search.best_estimator_
print(f"Best parameters found: {best_params}")

# Evaluate the best model
print("Evaluating the best SVM model...")
y_train_pred = best_svm_model.predict(X_train)
y_test_pred = best_svm_model.predict(X_test)
train_accuracy = accuracy_score(y_train, y_train_pred)
test_accuracy = accuracy_score(y_test, y_test_pred)
print(f"Train accuracy: {train_accuracy}")
print(f"Test accuracy: {test_accuracy}")

# Normalize confusion matrix
conf_matrix = confusion_matrix(y_test, y_test_pred)
conf_matrix_normalized = conf_matrix.astype('float') / conf_matrix.sum(axis=1)[:, np.newaxis]

# Plot confusion matrix
print("Plotting confusion matrix...")
sns.heatmap(conf_matrix_normalized, annot=True, fmt='.2f', cmap='Blues')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Normalized Confusion Matrix')
plt.show()

# Plot accuracies
print("Plotting accuracies...")
accuracies = [train_accuracy, test_accuracy]
labels = ['Train', 'Test']
colors = ['#1f77b4', '#ff7f0e']

plt.figure(figsize=(10, 6))
bars = plt.bar(labels, accuracies, color=colors)
plt.title('SVM Model Accuracy On Original Data')
plt.ylabel('Accuracy')
plt.ylim(0, 1)

# Add accuracy percentages below the bars
for bar, accuracy in zip(bars, accuracies):
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width() / 2.0, height - 0.05, f'{accuracy:.2%}', ha='center', va='bottom', color='white')

plt.show()

end_time = time.time()
print(f"Script execution complete. Total time taken: {end_time - start_time} seconds.")
