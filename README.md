Student Pass/Fail AI (Neural Network from Scratch in NumPy)

This project is a complete, functional Artificial Intelligence built from the ground up using only NumPy. It demonstrates the core mathematics and logic of deep learning without relying on any high-level libraries like TensorFlow or Keras.

The AI is trained to predict whether a student will "Pass" or "Fail" based on two features: their Score and their Study Hours.

Network Architecture (2-4-2)
This network uses a simple Multi-Layer Perceptron (MLP) architecture with one hidden layer. The flow of information is as follows:

[Input Layer] (2 Neurons)
     ( )  [Score]
     ( )  [Study Hours]
      |
      |  (Weights: 2x4)
      v
[Hidden Layer] (4 Neurons)
     ( )
     ( )
     ( )
     ( )
      |
      |  (Weights: 4x2)
      v
[Output Layer] (2 Neurons)
     ( )  [FAIL]
     ( )  [PASS]
Features
100% NumPy: The entire neural network (forward pass, backpropagation, and loss calculation) is built from scratch.

Data Generation: Includes a script using pandas to generate a 1,000-sample training dataset (student_data_1000.csv) with logical patterns.

Full Training Loop: Implements a complete "study" cycle:

Forward Pass: The AI makes a guess.

Loss Calculation: It measures how "wrong" its guess is (Cross-Entropy Loss).

Backward Pass (Backpropagation): It uses the error to "nudge" all its weights and biases, allowing it to learn from its mistakes.

Modern Architecture: Uses a sigmoid activation function for the hidden layer and a softmax function for the final output, which is standard for classifiers.

Live Predictions: After training, the script will prompt you to enter new student data and will provide a live, educated prediction with a confidence score.

How It Works
Data Generation: The script first creates a student_data_1000.csv file with 1,000 samples (650 "Pass" and 350 "Fail").

Training: The AI loads this data and trains on it for 1,000 epochs, adjusting its internal "knobs" (weights and biases) to get better at finding the patterns.

Prediction: Once trained, it asks for new, unseen data (a student's score and study hours) and uses its "educated" brain to predict the outcome.
