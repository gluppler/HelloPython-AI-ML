# 🧠 HelloPython-AI-ML

> A growing collection of small, well-documented AI & Machine Learning projects built from the ground up — starting with pure **NumPy**, expanding into **OpenCV**, **Matplotlib**, **PyTorch**, and **Keras**.

---

## 🌱 Overview

**HelloPython-AI-ML** is a learning-oriented repository focused on building and understanding machine learning, deep learning, and computer vision concepts from scratch — not just using prebuilt models.

This repo begins with **bare-metal implementations using NumPy** for foundational math and logic, then gradually evolves toward using **OpenCV**, **Matplotlib**, **PyTorch**, and **Keras** for practical AI and ML workflows.

The idea is simple:
- **Start small, build deep understanding.**
- **Evolve your codebase as your skills grow.**

---

## 🧩 Roadmap

| Stage | Focus | Libraries | Description |
|-------|--------|------------|--------------|
| **Phase 1** | Core Math & ML Foundations | `numpy`, `matplotlib` | Linear algebra, data manipulation, and simple visualizations. |
| **Phase 2** | Image Processing & Computer Vision | `opencv-python`, `numpy` | Working with image data, filters, edge detection, etc. |
| **Phase 3** | Neural Networks from Scratch | `numpy` | Implement perceptrons, feed-forward nets, and backprop manually. |
| **Phase 4** | Deep Learning Frameworks | `pytorch`, `keras` | Move from theory to scalable implementations. |
| **Phase 5** | Applied Projects | mixed | Small projects: digit recognition, image classification, clustering, etc. |

---

## 📁 Repository Structure

```

HelloPython-AI-ML/
│
├── README.md               <- This file
├── CONTRIBUTING.md         <- Guidelines for contributors
├── requirements.txt        <- Dependencies list
│
├── numpy_basics/           <- NumPy-only foundational projects
│   ├── linear_algebra.py
│   ├── simple_regression.py
│   └── matrix_operations.py
│
├── opencv_projects/        <- Image and video processing demos
│   ├── edge_detection.py
│   ├── color_space_conversion.py
│   └── object_tracking.py
│
├── matplotlib_visuals/     <- Plotting and data visualization examples
│   ├── loss_curve_demo.py
│   └── scatter_clusters.py
│
├── pytorch_learning/       <- Intro ML/DL projects using PyTorch
│   ├── linear_regression_pytorch.py
│   ├── cnn_mnist.py
│   └── autoencoder_demo.py
│
└── keras_learning/         <- Experiments using TensorFlow/Keras
├── sequential_basics.py
└── image_classifier.py

````

---

## 🧠 Example: Simple NumPy Project

```python
# numpy_basics/simple_regression.py

import numpy as np
import matplotlib.pyplot as plt

# Generate random linear data
X = np.linspace(0, 10, 100)
y = 3 * X + 4 + np.random.randn(100) * 2

# Linear regression using closed-form solution
X_b = np.c_[np.ones((100, 1)), X]  # Add bias term
theta_best = np.linalg.inv(X_b.T @ X_b) @ X_b.T @ y

# Predictions
y_pred = X_b @ theta_best

plt.scatter(X, y, label="Data")
plt.plot(X, y_pred, color="red", label="Best Fit")
plt.legend()
plt.title("Simple Linear Regression (NumPy)")
plt.show()
````

---

## 🚀 Getting Started

### 1️⃣ Clone the repository

```bash
git clone https://github.com/gluppler/HelloPython-AI-ML.git
cd HelloPython-AI-ML
```

### 2️⃣ Create a virtual environment

```bash
python -m venv venv
source venv/bin/activate   # On Windows use: venv\Scripts\activate
```

### 3️⃣ Install dependencies

```bash
pip install -r requirements.txt
```

Example `requirements.txt`:

```
numpy
matplotlib
opencv-python
torch
torchvision
tensorflow
keras
```

---

## 🧑‍💻 Contribution

Contributions are welcome!
To contribute:

1. Fork the repo
2. Create a new branch (`feature/new-idea`)
3. Commit your changes
4. Open a Pull Request

Read [`CONTRIBUTING.md`](CONTRIBUTING.md) for full guidelines.

---

## 🎯 Goals

* Build intuition for AI/ML concepts through **minimal, self-contained examples**
* Transition from **NumPy-only logic** → to **PyTorch/Keras abstractions**
* Provide a **hands-on playground** for AI learners, students, and engineers

---

## 📚 Learning References

* [NumPy Documentation](https://numpy.org/doc/stable/)
* [OpenCV Documentation](https://docs.opencv.org/)
* [Matplotlib Documentation](https://matplotlib.org/stable/)
* [PyTorch Tutorials](https://pytorch.org/tutorials/)
* [Keras Guide](https://keras.io/guides/)

---

## 🧙‍♂️ Author

**[Gabe Chew (gluppler)](https://github.com/gluppler)**
Software & Security Engineer | AI/ML Enthusiast | Creator of Cult of the LOLCOW & Behelit Systems

> *"Understanding comes from building — not importing."*

---

## 🏷️ License

This project is licensed under the **GNU General Public License** – see the [LICENSE](LICENSE) file for details.


