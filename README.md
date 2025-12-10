# Battery Curve Reconstruction & SOH Prediction Pipeline  
### CC (Constant-Current) + IC (Incremental Capacity) Full Deep Learning Pipeline  

---

# English Overview

This repository provides a complete machine-learning pipeline for:

- **CC Curve Reconstruction** (Residual CNN + Transformer)
- **IC Curve Generation & Reconstruction** (Gaussian smoothing + Transformer)
- **SOH (State of Health) Calculation**
- **SOH Prediction Models** (CNN + LSTM hybrid)
- Modularized scripts for CC and IC workflows

It is designed for research use on NASA battery aging datasets and supports easy extension to other batteries.

---

# 📁 Project Structure
.
├── cc_preprocess.py
├── cc_reconstruct_model.py
├── cc_soh_preprocess.py
├── cc_soh_model.py
├── cc_main.py
│
├── ic_preprocess.py
├── ic_reconstruct_model.py
├── ic_soh_preprocess.py
├── ic_soh_model.py
├── ic_main.py
│
├── data/ # Raw NASA battery data (must download separately)
└── README.md


---

# 📦 NASA Battery Dataset (IMPORTANT)

This repository **does not include** the raw NASA battery dataset.  
You must download it manually:

🔗 NASA Battery Aging Dataset  
[https://data.nasa.gov/dataset/Battery-Aging-Dataset/uj5r-zjdb](https://data.nasa.gov/dataset/li-ion-battery-aging-datasets)

Place the files (e.g., `B0005.csv`, `B0006.csv`, …) into the `data/` directory.

---

# 🚀 CC Pipeline Overview

### **1) Preprocessing**
- Load CC cycle data  
- Extract input portion (e.g., 10%)  
- Normalize & split datasets  

### **2) CC Reconstruction Model**
- Residual CNN blocks + BatchNorm + ReLU  
- Optional Transformer layer  
- Hyperparameter search using KerasTuner  
- Output: Reconstructed full CC voltage curve (300 points)

### **3) SOH Merge**
- Merge CC reconstructed curves with SOH labels  
- Based on capacity retention or provided SOH data

### **4) SOH Prediction**
- CNN + LSTM hybrid model  
- Predicts SOH per cycle  
- Outputs final degradation curves

---

# ⚡ IC Pipeline Overview

### **1) IC Preprocessing**
- Gaussian smoothing on voltage  
- Compute dQ/dV using Gaussian derivative  
- Build long-format dataset (voltage, IC, cycle)

### **2) IC Reconstruction Model**
- Pure Transformer architecture  
- Input section: 3.90–4.00 V  
- Reconstructs the entire IC curve

### **3) IC → SOH Mapping**
- Compute SOH using capacity retention  
- Merge SOH labels per cycle  

### **4) SOH Prediction (IC-based)**
- CNN + LSTM model predicts SOH  
- Produces smooth degradation curves

---

# ▶️ How to Run

### Run the entire CC pipeline:

### Run the entire IC pipeline:



---

# 📊 Output Examples

- Reconstructed CC & IC curves  
- True vs Predicted SOH plots  
- CSV files per battery containing:
  - Full reconstructed curve  
  - Cycle index  
  - SOH  
  - Voltage array / IC curve  

---


---

# 📈 Results — Model Prediction Example

### 🔹 CC Curve Reconstruction + SOH Prediction (Cycle #5 Example)

<p align="center">
  <img src="images/cc_curve_soh_5.png" width="75%">
</p>

The figure above shows:
- **Reconstructed CC voltage curve** from partial inputs  
- **True vs Predicted SOH trend** for the chosen cycle  
- Demonstrates high agreement between measured and reconstructed features  

---




# 🙌 Contributors

- **Jeong-Yong Shin** — Research + Full implementation  
- Assisted by ChatGPT (AI pair-programming)




