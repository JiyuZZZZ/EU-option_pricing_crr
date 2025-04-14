# 📈 European Option Pricing – Binomial Tree (CRR) vs Black-Scholes
This project was completed as part of my MSc Financial Mathematics coursework.
This project implements a **European call option pricing engine** using the **Cox-Ross-Rubinstein (CRR) binomial tree model**, and compares its output with the **Black-Scholes analytical formula**. The goal is to explore numerical approximation methods in financial mathematics and analyze their accuracy.

---

## 🧠 What It Does

- Prompts the user to input key parameters:
  - Initial stock price `S₀`
  - Risk-free interest rate `r`
  - Volatility `σ`
  - Maturity `T`
  - Strike price `K`
  - Number of time steps `N`
- Calculates the option price using the **CRR binomial tree method**
- Compares it to the **Black-Scholes closed-form solution**
- Reports pricing differences and discusses numerical convergence

---

## 📦 Technologies Used

- Python 3
- `NumPy` for numerical calculations
- `SciPy` for Black-Scholes formula (`scipy.stats.norm`)
- `matplotlib` (optional) for visualizing convergence or error

---

## 🧪 Sample Output

---

## 🔧 How to Run

```bash
python option_pricing_crr.py
