# CONTRIBUTING\_EXAMPLES.md

*HelloPython-AI-ML — PyTorch, Data Mining & Colab-ready*

This document demonstrates **how to contribute to HelloPython-AI-ML** while following our **10 universal coding principles**.
It includes **PyTorch examples**, **data mining practices**, and **Colab-ready code**, designed for **scalability, reproducibility, and enterprise-grade maintainability**.

---

## 1. Clarity Over Cleverness

* Keep code **readable** and **explicit**.
* Example:

```python
# Colab-ready installation
# !pip install torch torchvision

import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# Explicit dataset preparation
transform = transforms.ToTensor()
train_dataset = datasets.MNIST(root="./data", train=True, download=True, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
```

---

## 2. Explicit Data Handling (Data Mining Principle)

* Always show **data preprocessing** steps.

```python
import pandas as pd

# Load CSV dataset explicitly
df = pd.read_csv("https://raw.githubusercontent.com/mwaskom/seaborn-data/master/iris.csv")

# Encode categorical labels
df["species_encoded"] = df["species"].astype("category").cat.codes
print(df.head())
```

---

## 3. Portability & Hardware Flexibility

* Code must run on **CPU, GPU, Colab, and local machines**.

```python
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)
```

---

## 4. Document Your Work

* Explain **why each step is done**, not just what is done.

```python
# Normalize MNIST images
# Improves training stability and convergence
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])
```

---

## 5. Testing & Validation

* Validate **outputs and shapes**.

```python
class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(28*28, 10)
    def forward(self, x):
        return self.fc(x.view(-1, 28*28))

model = Net()
x = torch.randn(1, 1, 28, 28)
out = model(x)
assert out.shape == (1, 10), "Model output shape mismatch!"
```

---

## 6. Consistency in Style

* Use **consistent naming** and repo structure:

```
/data        # Datasets
/models      # Model checkpoints
/examples    # Example scripts
/notebooks   # Jupyter/Colab notebooks
```

* Keep **variable and function names uniform** across files.

---

## 7. Future-Proofing

* Save **models and experiment configs** for later use.

```python
# Save model checkpoint
torch.save(model.state_dict(), "mnist_model.pth")

# Load later (Colab or local)
model.load_state_dict(torch.load("mnist_model.pth", map_location=device))
```

---

## 8. Scalability

* Design code to scale **from small to large datasets**, CPU or GPU.

```python
for epoch in range(3):
    for data, target in train_loader:
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        loss = criterion(model(data), target)
        loss.backward()
        optimizer.step()
```

---

## 9. Reproducibility

* Always **fix random seeds** to make results consistent.

```python
import numpy as np
import random

seed = 42
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)
```

---

## 10. Community & Collaboration

* Provide **notebooks and example scripts** for easy onboarding.

```markdown
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](
    https://colab.research.google.com/github/YOUR_USER/HelloPython-AI-ML/blob/main/notebooks/mnist_baseline.ipynb
)
```

---

## Bonus: Data Mining Guidelines

1. **Data Quality First**

   * Validate and clean datasets before training.

```python
# Check for missing values
print(df.isnull().sum())
```

2. **Reproducibility Always**

   * Log dataset version, preprocessing steps, and experiment seed.

3. **Scalable Pipelines**

   * Use **DataLoader** with batching and multiprocessing.

```python
train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True, num_workers=4, pin_memory=True)
```

---

✅ **This CONTRIBUTING\_EXAMPLES.md ensures that:**

* Contributors can **run and fine-tune models in Colab immediately**.
* Code is **enterprise-grade**, **maintainable**, and **future-proof** for local hardware or servers.
* All **10 principles + data mining rules** are followed consistently.

---
