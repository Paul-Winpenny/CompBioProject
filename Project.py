import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization, LSTM, Conv1D, MaxPooling1D, Flatten
from tensorflow.keras.optimizers import Adam, SGD
from tensorflow.keras.regularizers import l2

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




y_pred = second_network.predict(intermediate_output_test)
y_pred_classes = np.argmax(y_pred, axis=1)
y_true_classes = np.argmax(y_test, axis=1)  # Use the encoded labels for y_test

# Compute confusion matrix
cm = confusion_matrix(y_true_classes, y_pred_classes)

# Normalize the confusion matrix
cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

labels = ['e', 'h', '_']



# Function to create sliding windows
def create_sliding_windows(encoded_seqs, window_size=13):
    padding = np.zeros((window_size // 2, 21))  # Zero vectors for padding
    padded_seqs = np.vstack([padding, encoded_seqs, padding])
    windows = np.array([padded_seqs[i:i + window_size] for i in range(len(encoded_seqs))])
    return windows

# Assuming the presence of functions parse_data, encode_sequences, and encode_labels
# Load and process training data
seq_train, labels_train = parse_data("E:\BioProjCopy\CompBioProject\protein-secondary-structure.train.txt")
eseq_train = encode_sequences(seq_train)
elab_train = encode_labels(labels_train)
windowed_input_train = create_sliding_windows(np.array(eseq_train), window_size=13)

# Load and process testing data
seq_test, labels_test = parse_data("E:\BioProjCopy\CompBioProject\protein-secondary-structure.test.txt")
eseq_test = encode_sequences(seq_test)
elab_test = encode_labels(labels_test)
windowed_input_test = create_sliding_windows(np.array(eseq_test), window_size=13)

# Function to build the second model (cascade)
def build_second_model(input_shape):
    model = Sequential([
        Dense(40, activation='sigmoid', input_shape=(input_shape,)),  # Input shape matches the output of the first network
        Dense(3, activation='softmax')  # Same output configuration as the first network
    ])
    model.compile(optimizer=Adam(),  # Using Adam optimizer
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    return model

# Assuming the presence of the train_cascade function

# Build the first model (Convolutional Neural Network)
def build_improved_model(input_shape):
    model = Sequential([
        Conv1D(filters=64, kernel_size=3, activation='sigmoid', input_shape=(input_shape[0], input_shape[1]),
               kernel_regularizer=l2(0.01)),  # Single Conv1D layer with increased filters and L2 regularization
        BatchNormalization(),
        MaxPooling1D(pool_size=2),
        Flatten(),
        Dense(40, activation='sigmoid', kernel_regularizer=l2(0.01)),  # L2 regularization on Dense layer
        Dropout(0.3),  # Dropout to prevent overfitting
        Dense(3, activation='softmax')
    ])
    optim = Adam(learning_rate=0.0001)  # Optimizer with a conservative learning rate
    model.compile(optimizer=optim, loss='categorical_crossentropy', metrics=['accuracy'])
    return model

# Build the cascade architecture
def build_cascade_model(first_model, second_model):
    # Train the first model
    history1 = first_model.fit(windowed_input_train, elab_train, epochs=25, validation_data=(windowed_input_test, elab_test), batch_size=32)
    
    # Generate intermediate outputs from the first model
    intermediate_output = first_model.predict(windowed_input_train)
    
    # Train the second model using the outputs of the first model
    history2 = second_model.fit(intermediate_output, elab_train, epochs=25, validation_data=(first_model.predict(windowed_input_test), elab_test), batch_size=32)
    
    return history1, history2

# Build and train both models
better_first_model = build_improved_model(windowed_input_train.shape[1:])
better_second_model = build_second_model(better_first_model.output_shape[1])
better_history1, better_history2 = build_cascade_model(better_first_model, better_second_model)

# Evaluate the cascade architecture
intermediate_output_test = better_first_model.predict(windowed_input_test)
test_loss, test_accuracy = better_second_model.evaluate(intermediate_output_test, elab_test)
print("Test accuracy:", test_accuracy)
# Set up a figure to host two subplots vertically

# Predict using the second network (Improved Cascaded Network)
y_pred = second_network.predict(intermediate_output_test)  # Use intermediate_output_test from the cascade architecture
y_pred_classes = np.argmax(y_pred, axis=1)
y_true_classes = np.argmax(elab_test, axis=1)  # Use the encoded labels for elab_test

# Compute confusion matrix
cm = confusion_matrix(y_true_classes, y_pred_classes)

# Normalize the confusion matrix
cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

labels = ['e', 'h', '_']


#Test on unseen data sheets
#RS126Data, C : loops
#H : Helix
#E : Stand

# Load and process testing data
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

def create_sliding_windows(encoded_seqs, window_size=13):
    padding = np.zeros((window_size // 2, 21))
    padded_seqs = np.vstack([padding, encoded_seqs, padding])
    # Flatten each window slice to make it compatible with the model input
    windows = np.array([padded_seqs[i:i + window_size].flatten() for i in range(len(encoded_seqs))])
    return windows


seq, labels = parse_unseen_data("E:\BioProjCopy\CompBioProject\RS126.data.txt")
labels = convert_unseen_labels(labels)
eseq = encode_sequences(seq)
elab = encode_labels(labels)

# Create sliding windows
window_size = 13
windowed_input_test = create_sliding_windows(eseq, window_size)

# Assuming `first_network` and `second_network` are your original cascade models
# Assuming `better_first_model` and `better_second_model` are your improved cascade models

# Test the original cascade model
intermediate_output_test_orig = first_network.predict(windowed_input_test)
test_loss_orig, test_accuracy_orig = second_network.evaluate(intermediate_output_test_orig, elab)
print("Original Cascade Model Test accuracy on unseen data:", test_accuracy_orig)


def create_sliding_windows(encoded_seqs, window_size=13):
    padding = np.zeros((window_size // 2, 21))  # Zero vectors for padding
    padded_seqs = np.vstack([padding, encoded_seqs, padding])
    windows = np.array([padded_seqs[i:i + window_size] for i in range(len(encoded_seqs))])
    return windows

seq, labels = parse_unseen_data("E:\BioProjCopy\CompBioProject\RS126.data.txt")
labels = convert_unseen_labels(labels)
eseq = encode_sequences(seq)
elab = encode_labels(labels)

# Create sliding windows
window_size = 13
windowed_input_test = create_sliding_windows(eseq, window_size)

# Test the improved cascade model
intermediate_output_test_improved = better_first_model.predict(windowed_input_test)
test_loss_improved, test_accuracy_improved = better_second_model.evaluate(intermediate_output_test_improved, elab)
print("Improved Cascade Model Test accuracy on unseen data:", test_accuracy_improved)

# Function to plot confusion matrix
def plot_confusion_matrix(y_true, y_pred, title):
    cm = confusion_matrix(y_true, y_pred)
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    labels = ['e', 'h', '_']
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm_normalized, annot=True, fmt=".2f", cmap='Reds', cbar=False, xticklabels=labels, yticklabels=labels)
    plt.title(title)
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.show()

# Predict and plot confusion matrix for original cascade model
y_pred_orig = second_network.predict(intermediate_output_test_orig)
y_pred_classes_orig = np.argmax(y_pred_orig, axis=1)
y_true_classes = np.argmax(elab, axis=1)
plot_confusion_matrix(y_true_classes, y_pred_classes_orig, 'Original Cascade Model Normalized Confusion Matrix')

# Predict and plot confusion matrix for improved cascade model
y_pred_improved = better_second_model.predict(intermediate_output_test_improved)
y_pred_classes_improved = np.argmax(y_pred_improved, axis=1)
plot_confusion_matrix(y_true_classes, y_pred_classes_improved, 'Improved Cascade Model Normalized Confusion Matrix')

# Plotting bar chart of test accuracies
models = ['Original Cascade Model', 'Improved Cascade Model']
accuracies = [test_accuracy_orig, test_accuracy_improved]

plt.figure(figsize=(10, 6))
plt.bar(models, accuracies, color=['blue', 'green'])
plt.xlabel('Models')
plt.ylabel('Accuracy')
plt.title('Test Accuracy of Original vs Improved Cascade Models')
for i, v in enumerate(accuracies):
    plt.text(i, v + 0.01, f"{v:.2%}", ha='center', va='bottom', fontsize=12)
plt.ylim(0, 1)
plt.show()