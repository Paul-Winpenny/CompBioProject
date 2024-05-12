import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Embedding, Flatten
from tensorflow.keras.optimizers import Adam
import numpy as np

from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt
import seaborn as sns
tf.keras.utils.set_random_seed(32)
tf.config.experimental.enable_op_determinism()

def parse_data(file_path):
    sequences = []
    labels = []
 
    with open(file_path, "r") as file:
        for line in file:
            if line == "end\n" or line == "<end>":
                sequences.append("end")
                labels.append("end")
                #print("End of sequence")
            elif len(line) == 4:
                sequences.append(line[0]) # Amino acid
                labels.append(line[2]) # e, h, _
                          
            else:
                #print("Something else")
                #print(line)
                pass
        
    
    return sequences, labels

def encode_seqeuences(sequences):
    amino_acid_mapping = {
    'A': 0,  'R': 1,  'N': 2,  'D': 3,  'C': 4,
    'Q': 5,  'E': 6,  'G': 7,  'H': 8,  'I': 9,
    'L': 10, 'K': 11, 'M': 12, 'F': 13, 'P': 14,
    'S': 15, 'T': 16, 'W': 17, 'Y': 18, 'V': 19,
    'end': 20
    }
    encoded_seqs = []
    for amino_acid in sequences:
        encoding = [0]*21
        encoding[amino_acid_mapping[amino_acid]] = 1
        encoded_seqs.append(encoding)

    return encoded_seqs

def encode_labels(labels):
    label_mapping = {
    'e': 0, 'h': 1, '_': 2, 'end': 3
    }
    #print("encoding labels")
    encoded_labels = []
    for label in labels:
        encoded_label = [0,0,0]
        if label_mapping[label] != 3:
            encoded_label[label_mapping[label]] = 1
        encoded_labels.append(encoded_label)
    return encoded_labels

    
seq,labels = parse_data("protein-secondary-structure.train.txt")
#print(seq)
eseq = encode_seqeuences(seq)
#print(labels)
elab = encode_labels(labels)

#Confirm that the encoding is correct
foundVal = ""
count = 0
while foundVal != "end":
    foundVal = seq[count]
    fV = eseq[count]
    #print("Character",foundVal,"is encoded as",fV)
    
    count += 1

print("End found at:",count)
count = 0
foundVal = []
while foundVal != [0]*20 + [1]: # when it finds foundVal = [0]*21, it will stop
    foundVal = eseq[count]
    count += 1   
print("End found at:",count)
def build_model(input_shape):
    learning_rate = 0.001
    optim = Adam(learning_rate=learning_rate)
    model = Sequential([
        Dense(40, activation='sigmoid', input_shape=(input_shape,)),  # Hidden layer with 40 units
        Dense(3, activation='softmax')  # Output layer for the three types of secondary structures
    ])
    model.compile(optimizer=optim, loss='categorical_crossentropy', metrics=['accuracy'])
    return model
def create_sliding_windows(encoded_seqs, window_size=13):
    # Padding to handle boundaries
    padding = np.zeros((window_size // 2, 21))  # Zero vectors for padding
    padded_seqs = np.vstack([padding, encoded_seqs, padding])
    
    # Create sliding windows
    windows = np.array([padded_seqs[i:i + window_size].flatten() for i in range(len(encoded_seqs))])
    return windows

windowed_input = create_sliding_windows(np.array(eseq))


# Assuming elab is already one-hot encoded
X_train, X_test, y_train, y_test = train_test_split(windowed_input, elab, test_size=0.2, random_state=42)
# Build the model with the correct input shape
model = build_model(X_train.shape[1])
y_train = np.array(y_train)
y_test = np.array(y_test)

# Train the model
history = model.fit(X_train, y_train, epochs=100, validation_data=(X_test, y_test), batch_size=32)
test_loss, test_accuracy = model.evaluate(X_test, y_test)
print("Test accuracy:", test_accuracy)
# Plot training & validation accuracy values
plt.figure(figsize=(14, 6))
plt.suptitle('80/20 Train/Test Split. Test Accuracy: ' + str(round(test_accuracy*100,2)) + '%', fontsize=16, y=1.05, x=0.5, ha='center')
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.yticks([i / 20 for i in range(21)])  # 0, 0.1, 0.2, ..., 1.0

plt.ylim(0.4, 1)  # Adjust the y-axis limit to provide space at the top

# Plot training & validation loss values
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.yticks([i / 10 for i in range(11)])  # 0, 0.1, 0.2, ..., 1.0

plt.ylim(0, max(max(history.history['loss']), max(history.history['val_loss'])))  # Adjust y-axis limit dynamically

# Put test accuracy as a label on the plot
# Adjust layout to remove white space
plt.tight_layout()
plt.show()


def calculate_q3_score(y_true, y_pred):
    total = len(y_true)
    correct = sum(np.argmax(y_true[i]) == np.argmax(y_pred[i]) for i in range(total))
    return correct / total

#Give me a confusion matrix
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

y_pred = model.predict(X_test)
y_pred_labels = np.argmax(y_pred, axis=1)
y_test_labels = np.argmax(y_test, axis=1)


q3_score = calculate_q3_score(y_test, y_pred)
print("Q3 Score:", q3_score)
# Compute the normalized confusion matrix
cm = confusion_matrix(y_test_labels, y_pred_labels, normalize='true')

# Plotting the normalized confusion matrix
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt=".2f", cmap='Blues', xticklabels=['E', 'H', '_'], yticklabels=['E', 'H', '_'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Normalized Confusion Matrix')
plt.show()

