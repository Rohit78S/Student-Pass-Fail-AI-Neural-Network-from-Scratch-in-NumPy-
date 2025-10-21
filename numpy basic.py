import numpy as np
import pandas as pd

print("Generating 1000 sample dataset...")

np.random.seed(42)
dataset = []

# Generate 650 PASS samples (high scores 70-100)
for i in range(650):
    score = np.random.uniform(70, 100)
    hours = np.random.uniform(1, 20)
    label_fail = 0
    label_pass = 1
    dataset.append([score, hours, label_fail, label_pass])

# Generate 350 FAIL samples
# 200 with medium scores (40-60) + various hours
for i in range(200):
    score = np.random.uniform(40, 60)
    hours = np.random.uniform(1, 20)
    label_fail = 1
    label_pass = 0
    dataset.append([score, hours, label_fail, label_pass])

# 150 with low scores (20-40) + low hours
for i in range(150):
    score = np.random.uniform(20, 40)
    hours = np.random.uniform(0.5, 10)
    label_fail = 1
    label_pass = 0
    dataset.append([score, hours, label_fail, label_pass])

# Shuffle the dataset
np.random.shuffle(dataset)

# Create DataFrame and save to CSV
df = pd.DataFrame(dataset, columns=['Score', 'Study_Hours', 'Label_Fail', 'Label_Pass'])
df.to_csv('student_data_1000.csv', index=False)
print(f"Dataset saved to 'student_data_1000.csv' ({len(df)} samples)")
print(f"PASS samples: {(df['Label_Pass'] == 1).sum()}")
print(f"FAIL samples: {(df['Label_Fail'] == 1).sum()}\n")

print("Reading data from CSV...")
df = pd.read_csv('student_data_1000.csv')

# Extract inputs and labels
inputs = df[['Score', 'Study_Hours']].values
labels = df[['Label_Fail', 'Label_Pass']].values

print(f"Loaded {len(inputs)} samples from CSV")
print(f"Input shape: {inputs.shape}")
print(f"Labels shape: {labels.shape}\n")

input_size = 2
hidden_size = 4
output_size = 2

hidden_weights = np.random.randn(input_size, hidden_size) * 0.01
hidden_biases = np.zeros(hidden_size) * 0.01
output_weights = np.random.randn(hidden_size, output_size) * 0.01
output_biases = np.zeros(output_size) * 0.01

learning_rate = 0.1
epochs = 1000


names = ["student1", "student2"]

def sigmoid(x):
    return 1 / ( 1 + np.exp(-np.clip(x, -500, 500)))
def sigmoid_derivative(x):
    return x * (1 - x)
def softmax(x):
    exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
    return exp_x / np.sum(exp_x, axis=1, keepdims=True)

def forward_pass(data):
    hidden_inputs = np.dot(data, hidden_weights) + hidden_biases
    hidden_outputs = sigmoid(hidden_inputs)
    output_inputs = np.dot(hidden_outputs, output_weights) + output_biases
    output_probs = softmax(output_inputs)
    return hidden_outputs, output_probs, hidden_inputs
def backward_pass(data, labels, hidden_outputs, output_probs, hidden_inputs):
    global hidden_weights, hidden_biases, output_weights, output_biases
    num_samples = data.shape[0]
    output_error = output_probs - labels
    output_delta = output_error
    output_weights -= learning_rate * np.dot(hidden_outputs.T, output_delta) /  num_samples
    output_biases -= learning_rate * np.sum(output_delta, axis=0) / num_samples

    hidden_error = np.dot(output_delta, output_weights.T)
    hidden_delta = hidden_error * sigmoid_derivative(hidden_outputs)

    hidden_weights -= learning_rate * np.dot(data.T, hidden_delta) / num_samples
    hidden_biases -= learning_rate * np.sum(hidden_delta, axis=0) / num_samples

def calculate_loss(output_probs, labels):
    return -np.mean(np.sum(labels * np.log(output_probs + 1e-8), axis=1))

for epoch in range(epochs):
    hidden_outputs, output_probs, hidden_inputs = forward_pass(inputs)
    backward_pass(inputs, labels, hidden_outputs, output_probs, hidden_inputs)
    if (epoch + 1) % 200 == 0:
        loss_calculation = calculate_loss(output_probs, labels)
        print("Epoch:", epoch + 1, "Loss:", loss_calculation)

print("\nTraining Complete!\n")

hidden_outputs, output_probs, _ = forward_pass(inputs)
predictions = np.argmax(output_probs, axis=1)

for i in range(min(2, len(inputs))):
    verdict = "PASS" if predictions[i] == 1 else "FAIL"
    confidence = output_probs[i, predictions[i]] * 100

    print(f"Prediction for Student{i + 1}: {verdict}")
    print(f"(Confidence: {confidence:.1f}%)\n")
# PREDICTIONS ON USER INPUT
user_inputs = np.array([
    [float(input("Enter Score for Student1: ")), float(input("Enter Study Hours for Student1: "))],
    [float(input("Enter Score for Student2: ")), float(input("Enter Study Hours for Student2: "))]
])

user_hidden_outputs, user_output_probs, _ = forward_pass(user_inputs)
user_predictions = np.argmax(user_output_probs, axis=1)

for i in range(len(names)):
    verdict = "PASS" if user_predictions[i] == 1 else "FAIL"
    confidence = user_output_probs[i, user_predictions[i]] * 100
    print(f"Prediction for {names[i]}: {verdict}")
    print(f"(Confidence: {confidence:.1f}%)\n")