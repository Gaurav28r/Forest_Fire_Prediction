import streamlit as st
import pandas as pd
import numpy as np
import time
import psutil
import os
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, recall_score

st.set_page_config(
    page_title="Forest Fire Prediction & Hardware Profiler",
    page_icon="🔥",
    layout="wide"
)

#SYSTEM UTILITIES 
def get_process_memory():
    """Returns memory usage of current process in MB."""
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / (1024 * 1024)

#DATA INGESTION & CACHING
@st.cache_data
def load_and_preprocess_data():
    url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/forest-fires/forestfires.csv'
    df = pd.read_csv(url)
    
    # Binary classification target: 1 = Fire, 0 = No Fire
    df['fire_risk'] = (df['area'] > 0).astype(int)
    df = df.drop('area', axis=1)
    
    # One-hot encode month and day
    df = pd.get_dummies(df, columns=['month', 'day'], drop_first=True)
    
    X = df.drop('fire_risk', axis=1)
    y = df['fire_risk']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    return X, y, X_train, X_test, X_train_scaled, X_test_scaled, y_train, y_test, scaler

# Load dataset once
X, y, X_train, X_test, X_train_scaled, X_test_scaled, y_train, y_test, scaler = load_and_preprocess_data()

# --- HEADER SECTION ---
st.title("🔥 Forest Fire Risk Prediction & CPU Profiler")
st.markdown(
    "**Course Project:** Computer Organization & Architecture (COA) × Machine Learning  \n"
    "This application demonstrates how decision tree branching and vector algebra affect hardware latency, multi-core parallelism, and RAM consumption."
)
st.divider()

# Tabs setup
tab_benchmark, tab_predict, tab_arch = st.tabs([
    "⚙️ COA Hardware Benchmarking", 
    "🌲 Live Fire Risk Prediction", 
    "💻 Architectural Insights"
])

# Coa hardware benchmarking
with tab_benchmark:
    st.subheader("CPU Execution Latency & RAM Consumption Profiling")
    st.markdown(
        "Run the benchmark suite to measure training latency, inference speed, and memory deltas across multiple iterations."
    )
    
    col_ctrl, col_info = st.columns([1, 2])
    with col_ctrl:
        iterations = st.slider("Benchmark Iterations (Averaging)", min_value=1, max_value=10, value=5)
        run_btn = st.button("🚀 Run Hardware Benchmark", type="primary", use_container_width=True)
        
    with col_info:
        st.info(
            "Running multiple iterations calculates the arithmetic mean across runs to eliminate "
            "operating system jitter, dynamic CPU frequency scaling, and background process interference."
        )

    if run_btn:
        models = {
            "RF (Parallel)": RandomForestClassifier(n_estimators=100, n_jobs=-1, random_state=42),
            "RF (Sequential)": RandomForestClassifier(n_estimators=100, n_jobs=1, random_state=42),
            "SVM": SVC(kernel='rbf', random_state=42),
            "Gradient Boosting": GradientBoostingClassifier(n_estimators=100, random_state=42)
        }

        plot_names = []
        plot_train_times = []
        plot_infer_times = []
        plot_mem_deltas = []
        accuracies = []
        recalls = []

        progress_bar = st.progress(0)
        status_text = st.empty()

        for idx, (name, model) in enumerate(models.items()):
            status_text.text(f"Profiling {name} across {iterations} iterations...")
            
            t_train_list, t_infer_list, mem_list = [], [], []

            for _ in range(iterations):
                mem_before = get_process_memory()
                t0 = time.perf_counter()
                model.fit(X_train_scaled, y_train)
                t1 = time.perf_counter()
                mem_after = get_process_memory()

                t2 = time.perf_counter()
                preds = model.predict(X_test_scaled)
                t3 = time.perf_counter()

                t_train_list.append((t1 - t0) * 1000)
                t_infer_list.append((t3 - t2) * 1000)
                mem_list.append(abs(mem_after - mem_before))

            acc = accuracy_score(y_test, preds)
            rec = recall_score(y_test, preds)

            plot_names.append(name)
            plot_train_times.append(np.mean(t_train_list))
            plot_infer_times.append(np.mean(t_infer_list))
            plot_mem_deltas.append(np.mean(mem_list))
            accuracies.append(acc * 100)
            recalls.append(rec * 100)

            progress_bar.progress((idx + 1) / len(models))

        status_text.empty()
        progress_bar.empty()

        # Display Metrics Table
        results_df = pd.DataFrame({
            "Model Architecture": plot_names,
            "Accuracy (%)": [f"{a:.2f}" for a in accuracies],
            "Recall (%)": [f"{r:.2f}" for r in recalls],
            "Avg Train Latency (ms)": [f"{t:.2f}" for t in plot_train_times],
            "Avg Infer Latency (ms)": [f"{i:.2f}" for i in plot_infer_times],
            "Avg Memory Delta (MB)": [f"{m:.2f}" for m in plot_mem_deltas]
        })
        
        st.write("### Benchmark Results Summary")
        st.dataframe(results_df, use_container_width=True)

        # Single Winner Evaluation
        max_lat = max(plot_train_times)
        max_mem = max(plot_mem_deltas) if max(plot_mem_deltas) > 0 else 1.0

        scores = [
            (plot_train_times[i] / max_lat) + (plot_mem_deltas[i] / max_mem)
            for i in range(len(plot_names))
        ]
        winner_idx = int(np.argmin(scores))
        winner_name = plot_names[winner_idx]

        st.success(
            f"🏆 **Hardware Efficiency Winner: {winner_name}**  \n"
            f"Delivered minimal resource overhead: **{plot_train_times[winner_idx]:.2f} ms** latency and "
            f"**{plot_mem_deltas[winner_idx]:.2f} MB** RAM footprint."
        )

        # Charts
        st.write("### Visual Hardware Profiling")
        colors = ['#2ca02c' if i == winner_idx else '#b0bec5' for i in range(len(plot_names))]

        sns.set_theme(style="whitegrid")
        fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

        sns.barplot(x=plot_names, y=plot_train_times, ax=axes[0], palette=colors, hue=plot_names, legend=False)
        axes[0].set_title('Avg CPU Training Latency (ms)', fontweight='bold')
        axes[0].set_ylabel('Milliseconds (ms)')
        axes[0].tick_params(axis='x', rotation=15)

        sns.barplot(x=plot_names, y=plot_mem_deltas, ax=axes[1], palette=colors, hue=plot_names, legend=False)
        axes[1].set_title('Avg RAM Delta (MB)', fontweight='bold')
        axes[1].set_ylabel('Megabytes (MB)')
        axes[1].tick_params(axis='x', rotation=15)

        fig.suptitle(f"Optimal Edge Architecture: {winner_name}", fontsize=14, fontweight='bold', color='#2ca02c')
        plt.tight_layout()
        st.pyplot(fig)

#live prediction
with tab_predict:
    st.subheader("Simulate Weather Conditions for Fire Hazard Prediction")
    st.markdown("Adjust the meteorological parameters below to evaluate real-time model inference.")

    col1, col2, col3 = st.columns(3)
    with col1:
        temp = st.slider("Temperature (°C)", min_value=0.0, max_value=40.0, value=25.0, step=0.1)
        rh = st.slider("Relative Humidity (%)", min_value=10.0, max_value=100.0, value=40.0, step=1.0)
    with col2:
        wind = st.slider("Wind Speed (km/h)", min_value=0.0, max_value=35.0, value=12.0, step=0.5)
        rain = st.slider("Rainfall Outside (mm)", min_value=0.0, max_value=10.0, value=0.0, step=0.1)
    with col3:
        ffmc = st.slider("FFMC (Fine Fuel Moisture)", min_value=15.0, max_value=100.0, value=85.0, step=0.5)
        dmc = st.slider("DMC (Duff Moisture Code)", min_value=1.0, max_value=300.0, value=100.0, step=1.0)

    selected_model_name = st.selectbox(
        "Select Model for Inference:",
        ["Support Vector Machine (SVM)", "Random Forest (Parallel)", "Gradient Boosting"]
    )

    if st.button("Predict Fire Risk", type="primary"):
        # Synthesize full feature row using training dataset column schema
        input_data = pd.DataFrame(0.0, index=[0], columns=X.columns)
        input_data['temp'] = temp
        input_data['RH'] = rh
        input_data['wind'] = wind
        input_data['rain'] = rain
        input_data['FFMC'] = ffmc
        input_data['DMC'] = dmc
        input_data['DC'] = X['DC'].mean()
        input_data['ISI'] = X['ISI'].mean()

        input_scaled = scaler.transform(input_data)

        # Train target model on the fly for demonstration
        if "SVM" in selected_model_name:
            inf_model = SVC(kernel='rbf', probability=True, random_state=42)
        elif "Random Forest" in selected_model_name:
            inf_model = RandomForestClassifier(n_estimators=100, n_jobs=-1, random_state=42)
        else:
            inf_model = GradientBoostingClassifier(n_estimators=100, random_state=42)

        t_start = time.perf_counter()
        inf_model.fit(X_train_scaled, y_train)
        pred = inf_model.predict(input_scaled)[0]
        prob = inf_model.predict_proba(input_scaled)[0][1]
        t_end = time.perf_counter()

        pred_latency = (t_end - t_start) * 1000

        st.divider()
        res_col1, res_col2 = st.columns(2)
        with res_col1:
            if pred == 1:
                st.error(f"⚠️ **HIGH FIRE RISK DETECTED** (Probability: {prob*100:.1f}%)")
            else:
                st.success(f"✅ **NO FIRE HAZARD DETECTED** (Probability: {(1-prob)*100:.1f}%)")
        with res_col2:
            st.metric("Inference Latency", f"{pred_latency:.2f} ms")

# architecturaal insights
with tab_arch:
    st.subheader("Computer Organization & Architecture (COA) Concepts")
    
    st.markdown("""
    * **Amdahl's Law & Parallelism:**  
      Random Forest distributes tree construction independently across all available CPU threads (`n_jobs=-1`). When restricted to a single thread (`n_jobs=1`), latency increases proportionally to the serial overhead.
      
    * **The Sequential Bottleneck:**  
      Gradient Boosting builds decision trees iteratively where tree $N$ minimizes the residual loss of tree $N-1$. This sequential dependency prevents thread parallelization, leaving extra CPU cores idle.
      
    * **ALU Workload vs. Branch Prediction:**  
      Decision trees rely on thousands of conditional jumps (`IF/ELSE`), stressing the hardware **Branch Target Buffer (BTB)** and risking instruction pipeline flushes upon mispredictions. Support Vector Machines utilize dense floating-point matrix operations that execute efficiently inside the **Arithmetic Logic Unit (ALU)** with high L1/L2 cache locality.
    """)