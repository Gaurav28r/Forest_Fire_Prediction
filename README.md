# 🔥 Silicon & Smoke: Hardware Profiling of ML Models
[![Python](https://img.shields.io/badge/Python-3.8%2B-blue?logo=python&logoColor=white)]()
[![Streamlit](https://img.shields.io/badge/Streamlit-Framework-FF4B4B?logo=streamlit&logoColor=white)]()
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)]()

> **Live Application:** [Click here to view the Streamlit App](https://forestfireprediction-8m9rnyn9a7zaxadvstkftf.streamlit.app/)

## 📌 Project Overview
This project bridges the gap between **Data Science** and **Computer Organization & Architecture (COA)**. While predicting forest fires using the UCI Machine Learning Repository dataset, this application goes a step further by profiling the exact hardware execution costs (CPU Latency, Multi-core Parallelism, and RAM Consumption) of different machine learning algorithms.

The goal is to determine the most resource-efficient algorithm for deployment on remote, power-constrained edge devices (e.g., solar-powered IoT sensors in forests).

## 🚀 Features
* **Automated Hardware Benchmarking:** Run live, multi-iteration benchmarks measuring execution time and Resident Set Size (RSS) memory footprint.
* **Algorithmic Contrast:** Evaluates Random Forest (Parallel & Sequential), Support Vector Machines (SVM), and Gradient Boosting.
* **Live Inference Simulation:** Adjust meteorological parameters (Temperature, Wind, Humidity, FFMC) to see real-time fire hazard probabilities and inference latencies.
* **COA Architectural Insights:** Visualizes and explains concepts like Amdahl's Law, Sequential Bottlenecks, and ALU vs. Branch Prediction optimization.

## 🧠 COA Concepts Demonstrated
1. **Amdahl's Law & Parallel Scaling:** Proves execution speedup by comparing multi-threaded Random Forest execution (`n_jobs=-1`) against single-threaded limits (`n_jobs=1`).
2. **The Sequential Bottleneck:** Demonstrates how sequentially dependent algorithms (Gradient Boosting) leave modern multi-core processors idle.
3. **ALU vs. Branch Prediction Workloads:** Contrasts the CPU pipeline-stressing `IF/ELSE` branch logic of Decision Trees against the highly cache-efficient, ALU-saturating matrix algebra of SVMs.

## 🏆 Final Hardware Verdict
Based on the benchmarks, the **Support Vector Machine (SVM)** proved to be the ultimate architecture for edge deployment. It delivers the highest predictive recall while maintaining a near-zero RAM footprint (~0.18 MB) and executing in a fraction of the time (~21 ms) compared to ensemble tree models.

## 💻 Local Installation and Execution

1. **Clone the repository:**
   ```bash
   git clone https://github.com/Gaurav28r/Forest_Fire_Prediction.git
   cd Forest_Fire_Prediction
   ```

2. **Install the required dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the Streamlit application:**
   ```bash
   streamlit run app.py
   ```

## 👨‍💻 Author
**Gaurav Verma**  
*B.Tech in CSE*  
*Rai Technology University, Bengaluru*
