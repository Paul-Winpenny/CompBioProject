import tensorflow as tf
from transformers import TFAutoModelForTokenClassification, AutoTokenizer
import numpy as np
from sklearn.model_selection import train_test_split

# Ensure TensorFlow and Transformers are up-to-date
print("TensorFlow version:", tf.__version__)
#print("Transformers version:", transformers.__version__)

# Load tokenizer and model
tokenizer = AutoTokenizer.from_pretrained("Rostlab/prot_bert", do_lower_case=False)
model = TFAutoModelForTokenClassification.from_pretrained("Rostlab/prot_bert", num_labels=3, from_pt=True)

def parse_data(file_path):
    with open(file_path, "r") as file:
        data = file.read().strip().split('\n')
    # Separate sequences and labels
    sequences = data[0::2]
    labels = data[1::2]
    return sequences, labels

def tokenize_and_encode(sequences, labels):
    # Tokenize all sequences
    tokenized_input = tokenizer(sequences, padding=True, truncation=True, return_tensors="tf", max_length=512)
    # Map labels C, H, E to numeric classes 0, 1, 2
    label_mapping = {'C': 0, 'H': 1, 'E': 2}
    label_ids = [[label_mapping[char] for char in label] for label in labels]
    padded_labels = tf.keras.preprocessing.sequence.pad_sequences(label_ids, padding='post', value=-100, maxlen=tokenized_input['input_ids'].shape[1])
    return tokenized_input, padded_labels

# Load and prepare data
sequences, labels = parse_data("RS126.data.txt")
tokenized_input, padded_labels = tokenize_and_encode(sequences, labels)

# Split the data
input_ids_train, input_ids_test, labels_train, labels_test = train_test_split(
    tokenized_input['input_ids'], padded_labels, test_size=0.2, random_state=42
)

# Convert to tf.data.Dataset
train_dataset = tf.data.Dataset.from_tensor_slices(({'input_ids': input_ids_train}, labels_train)).batch(8)
test_dataset = tf.data.Dataset.from_tensor_slices(({'input_ids': input_ids_test}, labels_test)).batch(8)

# Compile the model
optimizer = tf.keras.optimizers.Adam(learning_rate=5e-5)
model.compile(optimizer=optimizer, loss=model.compute_loss, metrics=['accuracy'])

# Train the model
history = model.fit(train_dataset, epochs=3, validation_data=test_dataset)

# Evaluate the model
print("Evaluation:")
model.evaluate(test_dataset)
