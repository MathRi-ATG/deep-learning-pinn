# PINN for Acid-Mediated Tumor Growth 🧠

This repository contains a TensorFlow implementation of a **Physics-Informed Neural Network** (**PINN**) to solve a system of nonlinear **partial differential equations** (**PDEs**) that model acid-mediated tumor growth. This project demonstrates how **PINNs** can solve complex biological models, even with sparse and noisy data, by embedding the governing physical laws directly into the neural network's training.

![PINN Concept](./Assets/concept.webp)
_Conceptual illustration: A standard neural network (magenta) might overfit sparse or noisy data. In contrast, a **PINN** (teal) is constrained by the underlying physics (dotted line), leading to a more accurate and physically plausible solution._

---

## Overview

**PINNs** are a modern, data-driven alternative to traditional PDE solvers. They are neural networks trained to satisfy not only observed data but also the governing differential equations themselves.

This project tackles the acid-mediated tumor growth (ATG) model, which is a set of coupled **PDEs** describing the interaction between normal cells, tumor cells, and the acidic environment they create.

The key components of our approach are:

- **Data Loss**: Measures the difference between the **PINN's** predictions and the observed training data.
- **Physics Loss**: Enforces adherence to the ATG partial differential equations.
- **Boundary/Initial Condition Loss**: Ensures the solution respects the problem's physical and temporal constraints.

---

## The Physical Model: Acid-Mediated Tumor Growth

The model, based on Gatenby and Gawlinski's work, describes the dynamics of normal cells ($N_n$), tumor cells ($N_t$), and excess H+ ion concentration ($C_h$) in a spherical geometry.

### Governing Equations

The system is described by the following set of **PDEs**:

1. **Normal Cells ($N_n$)**:

   $$\frac{\partial N_{n}}{\partial t} = r_{n1}N_{n}(1-\frac{N_{n}}{K_{n}}) - r_{n2}C_{h}N_{n}$$

   This models the logistic growth of normal cells, negatively impacted by excess H+ ions ($C_h$).

2. **Tumor Cells ($N_t$)**:

   $$\frac{\partial N_{t}}{\partial t} = r_{t1}N_{t}(1-\frac{N_{t}}{K_{t}}) + \frac{1}{r^{2}}\frac{\partial}{\partial r}\left[r^{2}D(N_{n})\frac{\partial N_{t}}{\partial r}\right]$$

   The tumor cell diffusivity $D(N_{n})$ depends on normal cell density:

   $$D(N_{n}) = D_{t}(1-\frac{N_{n}}{K_{n}})$$

   This describes logistic growth and nonlinear **diffusion** of tumor cells, slowed by higher normal cell density.

3. **Excess H+ Concentration ($C_h$)**:

   $$\frac{\partial C_{h}}{\partial t} = r_{h1}N_{t} - n_{h2}C_{h} + D_{h}\frac{1}{r^{2}}\frac{\partial}{\partial r}(r^{2}\frac{\partial C_{h}}{\partial r})$$

   This models H+ ion production by tumor cells, natural clearance, and **diffusion** through tissue.

---

## PINN Implementation ⚙️

We built a **PINN** to solve this system, combining data-driven learning with the physics-based equations above.

### 1. Network Architecture

- **Network**: A Multi-Layer Perceptron (MLP) with 3 hidden layers and 80 neurons per layer.
- **Activation Functions**: `tanh` for hidden layers and a `sigmoid` output to ensure positive concentrations.
- **Input/Output Scaling**: Time and space inputs are normalized, and outputs are scaled to physical units for loss calculation.

### 2. Loss Function

The total loss is a weighted sum of four components:

- **Data Loss**: Mean Squared Error (MSE) between **PINN** predictions and noisy training data.
- **Physics Loss**: Residuals of the three governing **PDEs**, evaluated at random collocation points.
- **Boundary Condition Loss**: Enforces Neumann (zero-gradient) conditions at the tumor center ($r=0$) and outer boundary ($r=0.5$ cm).
- **Initial Condition Loss**: Ensures predictions at $t=0$ match the initial state.

### 3. Training Strategy

- **Optimizer**: Adam optimizer with a learning rate of $1 \times 10^{-3}$.
- **Epochs**: Trained for 10,000 epochs.
- **Loss Weighting**: Weights balance loss terms (`500` for data, `50` for IC/BC, and `1` for PDE loss).
- **Automatic Differentiation**: Gradients computed exactly using TensorFlow's `tf.GradientTape`.

---

### Time and Space Complexity

- **Time Complexity**: $O(\text{epochs} \cdot (N_{\text{data}} + N_{\text{collocation}}) \cdot L \cdot W^2)$
- **Space Complexity**: $O(W^2 + n \cdot n_{\text{out}})$

---

### Expected Output

The script will:

1. Print the progress of the numerical solver.
2. Save noisy training data to `tumor_growth_data.csv`.
3. Print **PINN** training progress every 200 epochs.
4. Print final L2 errors for each variable.
5. Display and save a plot named `pinn_comparison.png` comparing results.

---

## Results 📊

![Results](./Assets/pinn_comparison.png)

The final plot demonstrates the **PINN's** ability to learn system dynamics, filter noise from training data, and closely match the true numerical solution.

_Example output plot comparing the PINN prediction (red dashed line) against the clean numerical solution (blue solid line) and noisy data points (black dots) used for training._

---

## References

For more on **Physics-Informed Neural Networks**, see:

- **Main PINN Website:** [https://maziarraissi.github.io/PINNs/](https://maziarraissi.github.io/PINNs/)
- Raissi, Maziar, et al. "[Physics-informed neural networks: A deep learning framework...](https://www.sciencedirect.com/science/article/pii/S0021999118307125)." _Journal of Computational Physics_ 378 (2019): 686-707.
- Raissi, Maziar, et al. "[Physics Informed Deep Learning (Part I)...](https://arxiv.org/abs/1711.10561)." _arXiv preprint arXiv:1711.10561_ (2017).
- Raissi, Maziar, et al. "[Physics Informed Deep Learning (Part II)...](https://arxiv.org/abs/1711.10566)." _arXiv preprint arXiv:1711.10566_ (2017).
