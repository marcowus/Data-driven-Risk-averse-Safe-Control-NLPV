# Data-Driven Risk-Averse Safe Control: Minimum-Variance Design

This repository contains Python code and simulation setups for our paper:

📄 **Title**: Data-Driven Risk-Averse Safe Control for Nonlinear Parameter-Varying Systems
📅 **Published in**: IEEE Control Systems Letters, vol. 8, pp. 2649–2655, 2024  
🔗 [DOI: 10.1109/LCSYS.2024.3509813](https://doi.org/10.1109/LCSYS.2024.3509813)

---

## 🧠 Abstract

This letter presents a data-driven risk-averse control strategy for stochastic nonlinear systems. The presented controller consists of two components: 1) A gain-scheduling control to ensure the invariance of the safe set by leveraging the concept of λ -contractivity, and 2) a nonlinear control to mitigate or eliminate the impact of the system’s nonlinearities on set invariance. To formalize a data-driven approach, closed-loop data-driven representations of both the gain scheduling and nonlinear dynamics are provided. These representations are then leveraged to impose λ -contractivity using a semidefinite programming (SDP) problem. The decision variables are optimized to minimize the variance of the system’s state distribution, ensuring that the system remains within the safe set with a specified probability. Additional constraints are incorporated to ensure robustness against unmeasured noise and residual nonlinearities. Simulation result validates the proposed approach, demonstrating its potential to enhance both safety and performance in stochastic nonlinear systems by managing gain scheduling, nonlinear dynamics, and uncertainty in a risk-aware manner.

---

## 🎯 Overview

Key contributions:
- **Data-driven representation**: Constructs closed-loop models from measured input–output data, capturing both gain‑scheduling dynamics and residual nonlinearities.
- **Gain scheduling via λ‑contractivity**: Enforces set invariance by imposing λ‑contractivity constraints on the scheduling policy.
- **Nonlinear compensation**: Designs a nonlinear corrective action to mitigate the impact of system nonlinearities on safety.
- **Risk-averse**: Implements the minimum-variance formulation to proactively reduce safety violation variability. 

---

## 🛠 Requirements

- Python 3.7 or newer  
- CVXPy  
- NumPy  
- SciPy  
- Matplotlib
- Mosek

---

## ▶️ Usage

1. **Clone** the repo:
   ```bash
   git clone https://github.com/Babak-Esmaeili/Data-driven-Risk-averse-Safe-Control-NLPV.git
   cd Data-driven-Risk-averse-Safe-Control-NLPV
   ```
2. **Run simulation for control design and visualization:**:
   ```bash
   python Codes/NLPV_MV_sim.py
   ```

---

## 📜 License & Contact

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.  
For questions or collaboration, contact:  
- **Babak Esmaeili** – esmaeil1@msu.edu 

---

## 📚 Citation

If you use this repository, please cite:
```bibtex
@article{esmaeili2024data,
  title={Data-Driven Risk-Averse Safe Control for Nonlinear Parameter-Varying Systems},
  author={Esmaeili, Babak and Modares, Hamidreza},
  journal={IEEE Control Systems Letters},
  volume={8},
  pages={2649--2654},
  year={2024},
  doi={10.1109/LCSYS.2024.3509813}
}
```
