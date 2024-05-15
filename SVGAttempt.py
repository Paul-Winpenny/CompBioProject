import time
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.svm import SVC

def parse_unseen_data(file_path):
    '''
    There is 150 instance and every instance contain 2 line .
    1st line = primary sequence (amino acid)
    2nd line = secondary sequence (C,H,E)
    #FNKEQQNAFYEILHLPNLNEEQRNGFIQSLKDDPSQSANLLAE
    CCCCHHHHHHHHHCCCCCCCHHHHHHHHHHHHCCCCCCCCCCC'''
    sequences = []
    labels = []
 
    with open(file_path, "r") as file:
        # Read the file line by line
        lines = file.readlines()
        #print(len(lines)    )
        # Iterate over the lines, 2 at a time
        primary = True
        for line in lines:
            cleaned_line = line.strip()
            if primary:
                for charac in cleaned_line:
                    sequences.append(charac)
                #sequences.append("end")
                primary = False
            else:
                for charac in cleaned_line:
                    labels.append(charac)
                #labels.append("end")
                primary = True
    

    return sequences, labels

def convert_unseen_labels(labels):
    '''
    Convert unseen labels to numerical format.
    C : _
    H : h
    E : e
    '''
    label_mapping = {'C': '_', 'H': 'h', 'E': 'e', 'end': 'end'}
    converted_labels = [label_mapping[label] for label in labels]
    return converted_labels
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

#Test on unseen data
seq_test2, labels_test2 = parse_unseen_data("E:\BioProjCopy\CompBioProject\RS126.data.txt")

eseq_train = encode_sequences(seq_train)
elab_train = encode_labels(labels_train)
eseq_test = encode_sequences(seq_test)
elab_test = encode_labels(labels_test)

labels_test2 = convert_unseen_labels(labels_test2)
eseq_test2 = encode_sequences(seq_test2)
elab_test2 = encode_labels(labels_test2)


# Function to create sliding windows
def create_sliding_windows(encoded_seqs, window_size=13):
    padding = np.zeros((window_size // 2, 21))
    padded_seqs = np.vstack([padding, encoded_seqs, padding])
    windows = np.array([padded_seqs[i:i + window_size].flatten() for i in range(len(encoded_seqs))])
    return windows

print("Preparing windowed input for model...")
# Prepare windowed input for model
windowed_input_train = create_sliding_windows(np.array(eseq_train),window_size=13)
windowed_input_test = create_sliding_windows(np.array(eseq_test),window_size=13)
windowed_input_test2 = create_sliding_windows(np.array(eseq_test2),window_size=13)

X_train = windowed_input_train
y_train = elab_train
X_test = windowed_input_test
y_test = elab_test
X_test2 = windowed_input_test2
y_test2 = elab_test2

# Define and fit the SVM model with manually specified parameters
print("Defining and fitting the SVM model...")
svm_model = SVC(C=1, gamma='scale', kernel='rbf', probability=True)
fit_start_time = time.time()
svm_model.fit(X_train, y_train)
fit_end_time = time.time()
print(f"SVM model fit complete. Time taken: {fit_end_time - fit_start_time} seconds.")

# Evaluate the SVM model
print("Evaluating the SVM model...")
y_train_pred = svm_model.predict(X_train)
y_test_pred = svm_model.predict(X_test)
y_test_pred2 = svm_model.predict(X_test2)

train_accuracy = accuracy_score(y_train, y_train_pred)
test_accuracy = accuracy_score(y_test, y_test_pred)
test_accuracy2 = accuracy_score(y_test2, y_test_pred2)

#print(f"Train accuracy: {train_accuracy}")
print(f"Test data accuracy: {test_accuracy}")
print(f"Unseen data accuracy: {test_accuracy2}")

# Normalize confusion matrix
conf_matrix = confusion_matrix(y_test, y_test_pred)
conf_matrix_normalized = conf_matrix.astype('float') / conf_matrix.sum(axis=1)[:, np.newaxis]

# Plot confusion matrix
print("Plotting confusion matrix...")
sns.heatmap(conf_matrix_normalized, annot=True, fmt='.2f', cmap='Blues')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Normalized Confusion Matrix 1988')
plt.show()

conf_matrix2 = confusion_matrix(y_test2, y_test_pred2)
conf_matrix_normalized2 = conf_matrix2.astype('float') / conf_matrix2.sum(axis=1)[:, np.newaxis]
sns.heatmap(conf_matrix_normalized2, annot=True, fmt='.2f', cmap='Blues')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Normalized Confusion Matrix RS126')

# Plot accuracies
print("Plotting accuracies...")
accuracies = [test_accuracy,test_accuracy2]
labels = ['1988 Dataset', 'RS126 Dataset']
colors = ['#1f77b4', '#ff7f0e']

plt.figure(figsize=(10, 6))
bars = plt.bar(labels, accuracies, color=colors)
plt.title('SVM Model Accuracy')
plt.ylabel('Accuracy')
plt.ylim(0, 1)

# Add accuracy percentages below the bars
for bar, accuracy in zip(bars, accuracies):
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width() / 2.0, height - 0.05, f'{accuracy:.2%}', ha='center', va='bottom', color='white')

plt.show()

end_time = time.time()
print(f"Script execution complete. Total time taken: {end_time - start_time} seconds.")
