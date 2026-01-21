# 🧠 Neural Networks – Mathematical Foundations and Basics

This repository provides a structured and intuitive introduction to **Neural Networks**, with special focus on their **mathematical foundations**, architecture, and learning mechanisms. It is intended for students and beginners aiming to build a strong conceptual and practical base in Artificial Intelligence and Machine Learning.

---

## 📌 What is a Neural Network?

A Neural Network is a computational model inspired by the human brain, consisting of interconnected processing units called neurons. These neurons work collectively to approximate complex functions and identify patterns in data.

Neural Networks are central to modern AI applications such as computer vision, speech recognition, natural language processing, and autonomous systems.

---

## 📚 Repository Contents

This repository covers:

- Introduction to Artificial Neural Networks (ANN)
- Biological Inspiration
- Neural Network Architecture  
  - Input Layer  
  - Hidden Layer(s)  
  - Output Layer  
- Neurons, Weights, and Bias
- Activation Functions
- Forward Propagation
- Loss Functions
- Backpropagation Algorithm (Conceptual)
- Learning Process
- Simple Mathematical and Code Examples

---

## 🎯 Learning Objectives

- Understand the structure and working of Neural Networks  
- Learn the mathematical representation of neurons  
- Grasp how networks learn from data  
- Build foundations for Deep Learning and AI  
- Connect theory with practical implementation  

---

## 🧮 Mathematical Foundations

### 1. Neuron Model

A single neuron computes a weighted sum of inputs and passes it through an activation function:

\[
z = \sum_{i=1}^{n} w_i x_i + b
\]

\[
a = f(z)
\]

Where:  
- \(x_i\) = input features  
- \(w_i\) = weights  
- \(b\) = bias  
- \(f(z)\) = activation function  
- \(a\) = output of neuron  

---

### 2. Vectorized Form

For a layer with multiple neurons:

\[
\mathbf{z} = \mathbf{W}\mathbf{x} + \mathbf{b}
\]
\[
\mathbf{a} = f(\mathbf{z})
\]

Where:  
- \(\mathbf{W}\) = weight matrix  
- \(\mathbf{x}\) = input vector  
- \(\mathbf{b}\) = bias vector  

---

### 3. Activation Functions

Common activation functions:

**Sigmoid:**
\[
f(z) = \frac{1}{1 + e^{-z}}
\]

**ReLU:**
\[
f(z) = \max(0, z)
\]

**Tanh:**
\[
f(z) = \tanh(z)
\]

**Softmax:**
\[
f(z_i) = \frac{e^{z_i}}{\sum_{j} e^{z_j}}
\]

---

### 4. Loss (Cost) Functions

Loss functions measure the error between predicted and actual output.

**Mean Squared Error (MSE):**
\[
L = \frac{1}{n}\sum_{i=1}^{n} (y_i - \hat{y}_i)^2
\]

**Cross-Entropy Loss:**
\[
L = -\sum_{i} y_i \log(\hat{y}_i)
\]

---

### 5. Backpropagation (Gradient Descent)

Neural Networks learn by minimizing loss using gradient descent:

\[
w = w - \eta \frac{\partial L}{\partial w}
\]

\[
b = b - \eta \frac{\partial L}{\partial b}
\]

Where:  
- \(\eta\) = learning rate  
- \(L\) = loss function  

This process propagates error backward and updates weights accordingly.

---

## ⚙️ Tools & Technologies

- Python  
- NumPy  
- Matplotlib  
- Scikit-learn  
- TensorFlow / PyTorch (optional)

---

## 🚀 Getting Started

### Clone the repository
```bash
git clone https://github.com/your-username/neural-networks-basics.git
cd neural-networks-basics
