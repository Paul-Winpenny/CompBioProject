import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam,SGD
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.optimizers import Adam
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Sequential
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
    encoded_labels = np.zeros((len(labels), 3))  # No need for encoding 'end'
    for idx, label in enumerate(labels):
        if label != 'end':
            encoded_labels[idx, label_mapping[label]] = 1
    return encoded_labels

# Load and process training and test data
seq_train, labels_train = parse_data("protein-secondary-structure.train.txt")
seq_test, labels_test = parse_data("protein-secondary-structure.test.txt")

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

# Prepare windowed input for model
windowed_input_train = create_sliding_windows(np.array(eseq_train))
windowed_input_test = create_sliding_windows(np.array(eseq_test))

def build_first_network(input_shape):
    model = Sequential([
        Dense(40, activation='sigmoid', input_shape=(input_shape,)),  # Hidden layer with 40 units
        Dense(3, activation='softmax')  # Output layer for the three types of secondary structures
    ])
    model.compile(optimizer=SGD(),  # Stochastic gradient descent optimizer
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    return model
def build_second_network():
    model = Sequential([
        Dense(40, activation='sigmoid', input_shape=(3,)),  # Input shape matches the output of the first network
        Dense(3, activation='softmax')  # Same output configuration as the first network
    ])
    model.compile(optimizer=SGD(),  # Using the same optimizer as the first network
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    return model
def train_cascade(first_model, second_model, X_train, y_train, X_test, y_test):
    # Train the first model and save the history
    history1 = first_model.fit(X_train, y_train, epochs=50, validation_data=(X_test, y_test))
    
    # Generate intermediate outputs from the first model
    intermediate_output = first_model.predict(X_train)
    
    # Train the second model using the outputs of the first model as inputs, and save the history
    history2 = second_model.fit(intermediate_output, y_train, epochs=50, validation_data=(first_model.predict(X_test), y_test))
    
    return history1, history2

X_train = windowed_input_train
y_train = elab_train
X_test = windowed_input_test
y_test = elab_test

# Define and train models
first_network = build_first_network(input_shape=X_train.shape[1])
second_network = build_second_network()

# Train both models and get their histories
history1, history2 = train_cascade(first_network, second_network, X_train, y_train, X_test, y_test)

# Evaluate models
intermediate_output_test = first_network.predict(X_test)
test_loss, test_accuracy = second_network.evaluate(intermediate_output_test, y_test)
print("Test accuracy:", test_accuracy)

plt.figure(figsize=(7, 12))  # Adjusted figure size for vertical layout

# Subplot for the accuracy of second model
plt.subplot(2, 1, 1)
plt.plot(history2.history['accuracy'], label='Train Accuracy')
plt.plot(history2.history['val_accuracy'], label='Validation Accuracy')
plt.title('Model 2 Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(loc='upper left')

# Subplot for the loss of second model
plt.subplot(2, 1, 2)
plt.plot(history2.history['loss'], label='Train Loss')
plt.plot(history2.history['val_loss'], label='Validation Loss')
plt.title('Model 2 Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(loc='upper left')

plt.tight_layout()
plt.show()