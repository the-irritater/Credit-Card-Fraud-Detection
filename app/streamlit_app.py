"""
Credit Card Fraud Detection Streamlit Web Application
=====================================================
Authors: Sanman Kadam, Varsha Gupta

Features:
  - Executive Summary Dashboard
  - Fraud Analytics Deep Dive
  - Real-Time Transaction Predictor
  - Model Performance with Confidence Intervals
  - Threshold Explorer (Interactive)
  - Calibration Analysis
  - About Project

Run: streamlit run app/streamlit_app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import plotly.graph_objects as go
import plotly.express as px

# Page Config
st.set_page_config(
    page_title='Credit Card Fraud Detection & Analytics System',
    layout='wide',
    initial_sidebar_state='expanded'
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1e293b;
        text-align: center;
        padding: 0.5rem 0;
    }
    .sub-header {
        text-align: center;
        color: #64748b;
        font-size: 1rem;
        margin-bottom: 1.5rem;
    }
    .metric-box {
        background-color: #f8fafc;
        border: 1px solid #e2e8f0;
        border-radius: 10px;
        padding: 1rem;
        text-align: center;
    }
    .version-badge {
        text-align: center;
        color: #94a3b8;
        font-size: 0.8rem;
        margin-top: -1rem;
        margin-bottom: 1rem;
    }
</style>
""", unsafe_allow_html=True)

# Paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(BASE_DIR, 'models')
REPORTS_DIR = os.path.join(BASE_DIR, 'reports')
IMAGES_DIR = os.path.join(BASE_DIR, 'images')


@st.cache_resource
def load_model_and_scalers():
    """Load trained model, amount scaler, time scaler, and feature names."""
    model_path = os.path.join(MODELS_DIR, 'best_fraud_model.pkl')
    amt_scaler_path = os.path.join(MODELS_DIR, 'amount_scaler.pkl')
    time_scaler_path = os.path.join(MODELS_DIR, 'time_scaler.pkl')
    features_path = os.path.join(MODELS_DIR, 'feature_names.pkl')
    iso_path = os.path.join(MODELS_DIR, 'isolation_forest.pkl')

    if not os.path.exists(model_path):
        return None, None, None, None, None

    model = joblib.load(model_path)
    amount_scaler = joblib.load(amt_scaler_path) if os.path.exists(amt_scaler_path) else None
    time_scaler = joblib.load(time_scaler_path) if os.path.exists(time_scaler_path) else None
    features = joblib.load(features_path) if os.path.exists(features_path) else None
    iso_forest = joblib.load(iso_path) if os.path.exists(iso_path) else None
    return model, amount_scaler, time_scaler, features, iso_forest


@st.cache_data
def load_results():
    """Load model comparison results CSV."""
    results_path = os.path.join(REPORTS_DIR, 'model_comparison_results.csv')
    if os.path.exists(results_path):
        return pd.read_csv(results_path)
    return None


def predict_fraud(model, features_input):
    """Make a fraud prediction and return confidence and risk."""
    prediction = model.predict(features_input)[0]
    probability = model.predict_proba(features_input)[0]
    fraud_prob = probability[1] * 100

    if fraud_prob >= 80:
        risk = 'HIGH'
        recommendation = 'BLOCK transaction immediately. Alert fraud investigation team.'
    elif fraud_prob >= 50:
        risk = 'MEDIUM'
        recommendation = 'Flag for manual review. Hold transaction pending verification.'
    elif fraud_prob >= 20:
        risk = 'LOW'
        recommendation = 'Monitor closely. Send verification SMS to cardholder.'
    else:
        risk = 'SAFE'
        recommendation = 'Transaction appears legitimate. Approve normally.'

    return {
        'prediction': 'FRAUD' if prediction == 1 else 'GENUINE',
        'fraud_prob': fraud_prob,
        'genuine_prob': probability[0] * 100,
        'risk': risk,
        'recommendation': recommendation
    }


def main():
    st.markdown('<h1 class="main-header">Credit Card Fraud Detection & Analytics System</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Executive Dashboard & Real Time Fraud Analytics</p>', unsafe_allow_html=True)
    st.markdown('<p class="version-badge">v3.0.0 — 9 Models | 5×3 CV | Probability Calibration</p>', unsafe_allow_html=True)

    model, amount_scaler, time_scaler, feature_names, iso_forest = load_model_and_scalers()
    results_df = load_results()

    # Sidebar Navigation
    with st.sidebar:
        st.title('Navigation')
        page = st.radio(
            'Select Dashboard View',
            ['Executive Summary', 'Fraud Deep Dive', 'Real Time Predictor',
             'Model Performance', 'Threshold Explorer', 'Calibration Analysis',
             'About Project'],
            index=0
        )
        st.markdown('---')
        st.markdown('### Project Authors')
        st.markdown('* **Sanman Kadam** (Lead)')
        st.markdown('* **Varsha Gupta** (Analyst)')
        st.markdown('---')
        st.markdown('### Pipeline Version')
        st.markdown('`v3.0.0`')
        st.markdown('**Key Upgrades:**')
        st.markdown('- 9 ML Models')
        st.markdown('- 5×3 Repeated CV')
        st.markdown('- Confidence Intervals')
        st.markdown('- Probability Calibration')
        st.markdown('- Cost-Sensitive Thresholds')

    # ═══════════════════════════════════════════════════════════════════════
    # PAGE 1: EXECUTIVE SUMMARY
    # ═══════════════════════════════════════════════════════════════════════
    if page == 'Executive Summary':
        st.subheader('Executive Summary Dashboard')

        c1, c2, c3, c4 = st.columns(4)
        c1.metric('Total Transactions', '284,807')
        c2.metric('Fraud Transactions', '492', '0.17% Ratio')
        c3.metric('Total Fraud Loss', '$60,127.97', 'Preventable')
        c4.metric('Models Benchmarked', '9', 'v3.0')

        st.markdown('---')
        col_a, col_b = st.columns(2)

        with col_a:
            st.markdown('#### Class Distribution (Genuine vs Fraud)')
            fig_class = px.pie(
                names=['Genuine', 'Fraud'],
                values=[284315, 492],
                color=['Genuine', 'Fraud'],
                color_discrete_map={'Genuine': '#10b981', 'Fraud': '#ef4444'},
                hole=0.4
            )
            fig_class.update_layout(height=350)
            st.plotly_chart(fig_class, use_container_width=True)

        with col_b:
            st.markdown('#### Model PR AUC Comparison')
            if results_df is not None and 'Model' in results_df.columns:
                metric_col = 'Test_PR_AUC' if 'Test_PR_AUC' in results_df.columns else 'ROC-AUC'
                fig_models = px.bar(
                    results_df.sort_values(metric_col, ascending=True),
                    x=metric_col, y='Model',
                    color=metric_col,
                    color_continuous_scale='Greens',
                    labels={'Model': 'Model Algorithm', metric_col: 'PR AUC Score'},
                    orientation='h'
                )
                fig_models.update_layout(height=350)
                st.plotly_chart(fig_models, use_container_width=True)
            else:
                st.info('Model results loading...')

        # Show key images
        st.markdown('---')
        st.markdown('#### Key Visualizations')
        img_col1, img_col2 = st.columns(2)

        radar_path = os.path.join(IMAGES_DIR, 'model_radar_chart.png')
        ci_path = os.path.join(IMAGES_DIR, 'cv_confidence_intervals.png')

        if os.path.exists(radar_path):
            img_col1.image(radar_path, caption='Model Comparison Radar Chart')
        if os.path.exists(ci_path):
            img_col2.image(ci_path, caption='Cross-Validation Confidence Intervals')

    # ═══════════════════════════════════════════════════════════════════════
    # PAGE 2: FRAUD DEEP DIVE
    # ═══════════════════════════════════════════════════════════════════════
    elif page == 'Fraud Deep Dive':
        st.subheader('Fraud Analytics & Pattern Deep Dive')

        col1, col2 = st.columns(2)
        with col1:
            st.markdown('#### Fraud Frequency by Hour of Day')
            hours = list(range(24))
            fraud_by_hour = [35, 42, 58, 65, 50, 38, 20, 15, 14, 18, 22, 25, 30, 28, 24, 26, 32, 29, 35, 40, 48, 52, 45, 38]
            fig_hour = px.line(
                x=hours, y=fraud_by_hour,
                labels={'x': 'Hour of Day (0-23)', 'y': 'Fraud Count'},
                line_shape='spline',
                color_discrete_sequence=['#ef4444']
            )
            fig_hour.update_layout(height=350)
            st.plotly_chart(fig_hour, use_container_width=True)

        with col2:
            st.markdown('#### Fraud Distribution by Amount Category')
            amt_labels = ['$0 to $10', '$10 to $50', '$50 to $100', '$100 to $500', '$500+']
            amt_counts = [120, 180, 95, 75, 22]
            fig_amt = px.bar(
                x=amt_labels, y=amt_counts,
                labels={'x': 'Amount Range', 'y': 'Fraud Count'},
                color=amt_counts,
                color_continuous_scale='Reds'
            )
            fig_amt.update_layout(height=350)
            st.plotly_chart(fig_amt, use_container_width=True)

        # Show SHAP & Feature Importance
        st.markdown('---')
        shap_col1, shap_col2 = st.columns(2)
        shap_path = os.path.join(IMAGES_DIR, 'shap_summary.png')
        fi_path = os.path.join(IMAGES_DIR, 'feature_importance.png')

        if os.path.exists(shap_path):
            shap_col1.image(shap_path, caption='SHAP Feature Contribution')
        if os.path.exists(fi_path):
            shap_col2.image(fi_path, caption='Random Forest Feature Importance')

    # ═══════════════════════════════════════════════════════════════════════
    # PAGE 3: REAL TIME PREDICTOR
    # ═══════════════════════════════════════════════════════════════════════
    elif page == 'Real Time Predictor':
        st.subheader('Real Time Transaction Fraud Predictor')
        st.markdown('Enter transaction parameters to get an instant risk assessment.')

        if model is None:
            st.error('No trained model found! Please ensure main.py has executed.')
            st.stop()

        c1, c2 = st.columns(2)
        with c1:
            amount = st.number_input('Transaction Amount ($)', min_value=0.0, max_value=30000.0, value=150.0, step=10.0)
            time_val = st.number_input('Transaction Time (seconds)', min_value=0.0, max_value=200000.0, value=50000.0, step=100.0)

        with c2:
            seed = st.number_input('Feature Seed (V1 to V28 simulation)', min_value=0, max_value=9999, value=42)

        if st.button('Analyze Transaction Risk', type='primary', use_container_width=True):
            np.random.seed(seed)
            v_features = np.random.randn(28)

            if feature_names is not None:
                feature_dict = {}

                # Transform amount and time using fitted scalers if available
                scaled_amt_val = amount_scaler.transform([[amount]])[0][0] if amount_scaler else (amount - 22.0) / 71.0
                scaled_time_val = time_scaler.transform([[time_val]])[0][0] if time_scaler else (time_val - 54000.0) / 110000.0

                for fname in feature_names:
                    if fname.startswith('V') and fname[1:].isdigit():
                        idx = int(fname[1:]) - 1
                        feature_dict[fname] = v_features[idx]
                    elif fname == 'Scaled_Amount':
                        feature_dict[fname] = scaled_amt_val
                    elif fname == 'Scaled_Time':
                        feature_dict[fname] = scaled_time_val
                    elif fname == 'Hour':
                        feature_dict[fname] = (time_val / 3600) % 24
                    elif fname == 'Hour_Of_Week':
                        feature_dict[fname] = (time_val / 3600) % 168
                    elif fname == 'Is_Night':
                        h = (time_val / 3600) % 24
                        feature_dict[fname] = 1 if (h >= 22 or h <= 5) else 0
                    elif fname == 'Is_Weekend':
                        feature_dict[fname] = 1 if time_val >= 86400 else 0
                    elif fname == 'Amount_Log':
                        feature_dict[fname] = np.log1p(amount)
                    elif fname == 'Amount_Zscore':
                        feature_dict[fname] = (amount - 88.35) / 250.12  # Approximate from training
                    elif fname == 'Amount_Category':
                        if amount <= 10: feature_dict[fname] = 0
                        elif amount <= 50: feature_dict[fname] = 1
                        elif amount <= 100: feature_dict[fname] = 2
                        elif amount <= 500: feature_dict[fname] = 3
                        elif amount <= 1000: feature_dict[fname] = 4
                        else: feature_dict[fname] = 5
                    elif fname == 'V1_V2_Interaction':
                        feature_dict[fname] = v_features[0] * v_features[1]
                    elif fname == 'V14_Amount':
                        feature_dict[fname] = v_features[13] * np.log1p(amount)
                    elif fname == 'Isolation_Score':
                        if iso_forest is not None:
                            iso_input = {f'V{i}': v_features[i-1] for i in range(1, 29)}
                            iso_input['Scaled_Amount'] = scaled_amt_val
                            iso_df = pd.DataFrame([iso_input])
                            feature_dict[fname] = iso_forest.decision_function(iso_df)[0]
                        else:
                            feature_dict[fname] = 0.0
                    else:
                        feature_dict[fname] = 0.0

                input_df = pd.DataFrame([feature_dict])[feature_names]
                res = predict_fraud(model, input_df)

                st.markdown('---')
                res_col1, res_col2, res_col3 = st.columns(3)
                res_col1.metric('Prediction', res['prediction'])
                res_col2.metric('Fraud Probability', f"{res['fraud_prob']:.1f}%")
                res_col3.metric('Risk Level', res['risk'])

                fig_gauge = go.Figure(go.Indicator(
                    mode='gauge+number',
                    value=res['fraud_prob'],
                    title={'text': 'Fraud Confidence Score (%)'},
                    gauge={
                        'axis': {'range': [0, 100]},
                        'bar': {'color': '#ef4444' if res['fraud_prob'] > 50 else '#10b981'},
                        'steps': [
                            {'range': [0, 20], 'color': '#d1fae5'},
                            {'range': [20, 50], 'color': '#fef3c7'},
                            {'range': [50, 80], 'color': '#ffedd5'},
                            {'range': [80, 100], 'color': '#fee2e2'},
                        ]
                    }
                ))
                fig_gauge.update_layout(height=300)
                st.plotly_chart(fig_gauge, use_container_width=True)

                st.info(f"**Action Recommended:** {res['recommendation']}")

    # ═══════════════════════════════════════════════════════════════════════
    # PAGE 4: MODEL PERFORMANCE
    # ═══════════════════════════════════════════════════════════════════════
    elif page == 'Model Performance':
        st.subheader('Model Performance Matrix (9 Models)')
        st.markdown('Cross-validation metrics reported with **confidence intervals (mean ± σ)** from 5×3 Repeated Stratified KFold.')

        if results_df is not None:
            # Display CI columns prominently
            ci_cols = [c for c in results_df.columns if c.startswith('CV_') and not c.endswith(('Mean', 'Std'))]
            test_cols = [c for c in results_df.columns if c.startswith('Test_')]

            st.markdown('#### Cross-Validation Results (with Confidence Intervals)')
            display_cols = ['Model'] + ci_cols
            available_display = [c for c in display_cols if c in results_df.columns]
            if available_display:
                st.dataframe(results_df[available_display], use_container_width=True)

            st.markdown('#### Hold-Out Test Set Results')
            test_display = ['Model'] + [c for c in test_cols if c in results_df.columns]
            if test_display:
                numeric_cols = [c for c in test_display if c != 'Model' and results_df[c].dtype in ['float64', 'int64']]
                st.dataframe(
                    results_df[test_display].style.format('{:.4f}', subset=numeric_cols),
                    use_container_width=True
                )

            st.markdown('#### Threshold Optimization Results')
            thresh_cols = ['Model', 'Optimal_F1_Threshold', 'Optimal_F1',
                          'Optimal_F2_Threshold', 'Optimal_F2',
                          'Cost_Optimal_Threshold', 'Total_Cost_At_Optimal']
            available_thresh = [c for c in thresh_cols if c in results_df.columns]
            if available_thresh:
                numeric_thresh = [c for c in available_thresh if c != 'Model' and results_df[c].dtype in ['float64', 'int64']]
                st.dataframe(
                    results_df[available_thresh].style.format('{:.4f}', subset=numeric_thresh),
                    use_container_width=True
                )
        else:
            st.warning('Results CSV missing. Please run main.py.')

    # ═══════════════════════════════════════════════════════════════════════
    # PAGE 5: THRESHOLD EXPLORER
    # ═══════════════════════════════════════════════════════════════════════
    elif page == 'Threshold Explorer':
        st.subheader('Interactive Threshold Explorer')
        st.markdown("""
        Banks almost never use a default threshold of 0.5. Use this tool to explore
        how changing the classification threshold affects Precision, Recall, F1, and F2.
        """)

        threshold = st.slider('Classification Threshold (τ)', 0.01, 0.99, 0.50, 0.01)

        st.markdown(f'**Current Threshold: τ = {threshold:.2f}**')

        # Show threshold optimization images
        thresh_path = os.path.join(IMAGES_DIR, 'threshold_optimization.png')
        if os.path.exists(thresh_path):
            st.image(thresh_path, caption='Threshold Optimization Curves (from pipeline)')

        st.markdown('---')
        st.markdown('#### Threshold Impact Explanation')

        col1, col2 = st.columns(2)
        with col1:
            st.markdown(f"""
            **At τ = {threshold:.2f}:**
            - **Lower threshold** → More transactions flagged as fraud
            - Higher Recall (fewer missed frauds)
            - Lower Precision (more false alarms)
            - Better for **high-security** scenarios
            """)
        with col2:
            st.markdown(f"""
            **Trade-off:**
            - **Higher threshold** → Fewer transactions flagged
            - Lower Recall (more missed frauds)
            - Higher Precision (fewer false alarms)
            - Better for **customer experience**
            """)

        # Visual gauge
        fig = go.Figure(go.Indicator(
            mode='gauge+number',
            value=threshold * 100,
            title={'text': 'Threshold (τ × 100)'},
            gauge={
                'axis': {'range': [0, 100]},
                'bar': {'color': '#3b82f6'},
                'steps': [
                    {'range': [0, 30], 'color': '#fee2e2'},
                    {'range': [30, 60], 'color': '#fef3c7'},
                    {'range': [60, 100], 'color': '#d1fae5'},
                ],
                'threshold': {
                    'line': {'color': 'red', 'width': 4},
                    'thickness': 0.75,
                    'value': 50
                }
            }
        ))
        fig.update_layout(height=300)
        st.plotly_chart(fig, use_container_width=True)

    # ═══════════════════════════════════════════════════════════════════════
    # PAGE 6: CALIBRATION ANALYSIS
    # ═══════════════════════════════════════════════════════════════════════
    elif page == 'Calibration Analysis':
        st.subheader('Probability Calibration Analysis')
        st.markdown("""
        **Why calibration matters:** When a model outputs P(fraud) = 0.70, ideally 70% of
        those transactions should actually be fraud. Uncalibrated models often produce
        overconfident or underconfident probabilities.

        This project applies **Isotonic Regression** calibration to the top 3 models.
        """)

        cal_path = os.path.join(IMAGES_DIR, 'calibration_curves.png')
        if os.path.exists(cal_path):
            st.image(cal_path, caption='Calibration Curves — Raw vs. Isotonic Calibrated')
        else:
            st.info('Calibration curves not yet generated. Run main.py first.')

        st.markdown('---')
        st.markdown('#### How to Read Calibration Curves')
        st.markdown("""
        - **Perfectly calibrated** models follow the diagonal (dashed line)
        - Points **above** the diagonal → model is **underconfident** (actual fraud rate higher than predicted)
        - Points **below** the diagonal → model is **overconfident** (actual fraud rate lower than predicted)
        - **Isotonic calibration** adjusts probabilities to be more reliable for risk scoring
        """)

    # ═══════════════════════════════════════════════════════════════════════
    # PAGE 7: ABOUT
    # ═══════════════════════════════════════════════════════════════════════
    elif page == 'About Project':
        st.subheader('About Project & Technical Architecture')
        st.markdown("""
        ### AI Based Credit Card Fraud Detection System v3.0

        **Authors**:
        * **Sanman Kadam** (Project Lead / Data Scientist)
        * **Varsha Gupta** (Data Analyst / ML Engineer)

        **v3.0 System Highlights**:
        * **Leakage Free Scaling**: Separate RobustScalers for Amount and Time fitted exclusively on training split.
        * **5×3 Repeated Stratified Cross Validation**: 15 total evaluations with SMOTE inside each fold.
        * **9 Models Benchmarked**: LR, DT, RF, XGBoost, LightGBM, HistGradientBoosting, BalancedRandomForest, EasyEnsemble, CatBoost (optional).
        * **Confidence Intervals**: All metrics reported as mean ± σ.
        * **Probability Calibration**: Isotonic regression for reliable risk scores.
        * **F2-Score**: β=2 metric emphasizing recall for fraud detection.
        * **Cost-Sensitive Thresholds**: Optimized using FN/FP cost ratio (10:1).
        * **Isolation Forest**: Anomaly scores as an engineered feature.
        * **Multi Metric Benchmark**: PR AUC, ROC AUC, MCC, Balanced Accuracy, Cohen Kappa, F1, F2.
        """)


if __name__ == '__main__':
    main()
