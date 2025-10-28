
---

## 🚀 Implemented Optimization Techniques

### ✅ Adam Optimizer

Adam (Adaptive Moment Estimation) combines:

- **Momentum** (gradient moving average)
- **RMSProp** (squared gradient normalization)

This helps improve convergence speed and prevents oscillations.

| Function | Description |
|---------|-------------|
| `initialize_adam(parameters)` | Initializes `v` and `s` dictionaries for all layers |
| `update_parameters_with_adam(...)` | Updates parameters using Adam’s rule with bias correction |

---

### ✅ Learning Rate Decay

Two decay strategies included:

| Function | Strategy | Formula |
|---------|----------|---------|
| `update_learning_rate(lr0, epoch, decay_rate)` | Continuous decay | lr = lr0 / (1 + decay_rate × epoch) |
| `schedule_learning_rate_decay(lr0, epoch, decay_rate, interval)` | Step-based decay | lr = lr0 / (1 + decay_rate × ⌊epoch/interval⌋) |

These methods help stabilize late-stage training and reduce overshooting.

---

### ✅ Mini-Batch Gradient Descent Utility

| Function | Description |
|---------|-------------|
| `random_mini_batches(X, Y, mini_batch_size)` | Shuffles dataset and splits into mini-batches |

Advantages:

- Faster convergence than full-batch GD
- Lower variance than stochastic GD
- Supports vectorized computation

---
