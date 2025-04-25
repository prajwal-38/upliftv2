import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import shap

def load_simulator_models():
    treated_model = joblib.load('models/treated_model.pkl')
    control_model = joblib.load('models/control_model.pkl')
    feature_cols = joblib.load('models/feature_cols.pkl')
    scaler = joblib.load('models/scaler.pkl')
    feature_stats = joblib.load('models/feature_stats.pkl')  #min, max, mean, std for each feature
    return treated_model, control_model, feature_cols, scaler, feature_stats

def run_simulator():
    st.title("Conversion Uplift Simulator")
    
    treated_model, control_model, feature_cols, scaler, feature_stats = load_simulator_models()
    
    #Create base features without interactions
    base_features = [col for col in feature_cols if '_interaction' not in col]
    
    #Create sidebar for user inputs
    st.sidebar.header("User Profile")
    
    #Create user profile with sliders
    user_profile = {}
    for feature in base_features:

        if feature in feature_stats:
            stats = feature_stats[feature]
            min_val = float(stats['min'])
            max_val = float(stats['max'])
            mean_val = float(stats['mean'])

            user_profile[feature] = st.sidebar.slider(
                f"{feature}", 
                min_value=min_val,
                max_value=max_val,
                value=mean_val
            )

    for i, feat1 in enumerate(base_features):
        for feat2 in base_features[i+1:]:
            interaction_name = f"{feat1}_{feat2}_interaction"
            if interaction_name in feature_cols:
                user_profile[interaction_name] = user_profile[feat1] * user_profile[feat2]

    user_df = pd.DataFrame([user_profile])
    

    treated_prob = treated_model.predict_proba(user_df)[0, 1]
    control_prob = control_model.predict_proba(user_df)[0, 1]
    uplift = treated_prob - control_prob
    

    st.header("Conversion Probability")
    col1, col2, col3 = st.columns(3)
    col1.metric("If Treated", f"{treated_prob:.2%}")
    col2.metric("If Not Treated", f"{control_prob:.2%}")
    col3.metric("Uplift", f"{uplift:.2%}")


    try:
        st.header("Feature Contribution to Uplift (SHAP)")
        
        # Explain both models
        treated_explainer = shap.TreeExplainer(treated_model)
        control_explainer = shap.TreeExplainer(control_model)
        
        treated_shap_values = treated_explainer.shap_values(user_df)
        control_shap_values = control_explainer.shap_values(user_df)
        
        # For classification, get SHAP for the positive class (class 1)
        if isinstance(treated_shap_values, list):
            treated_shap_values = treated_shap_values[1]
        if isinstance(control_shap_values, list):
            control_shap_values = control_shap_values[1]
            
        # Calculate Uplift SHAP for the single user instance
        # uplift_shap = treated_shap - control_shap
        # Expected value for uplift: E[treatment_pred] - E[control_pred]
        # Note: SHAP library's force plot needs a single expected value.
        # Calculating a direct uplift force plot is complex as it involves two models.
        # A common workaround is to show the treatment model's explanation,
        # as it often drives the decision, but acknowledge it's not direct uplift explanation.
        # OR we can plot the difference in SHAP values as a bar chart.

        # Option 1: Show Treatment Force Plot (as before, but clarify)
        st.write("SHAP Force Plot (Explaining Treatment Model Prediction)")
        st.caption("This plot shows factors driving the prediction *if treated*. It doesn't directly show uplift contribution.")
        plt.figure(figsize=(10, 4))
        # Use expected_value[1] if it's a list (for multi-class)
        expected_value_treatment = treated_explainer.expected_value[1] if isinstance(treated_explainer.expected_value, (list, np.ndarray)) else treated_explainer.expected_value
        shap.plots.force(
            expected_value_treatment, 
            treated_shap_values[0], # SHAP values for the first (and only) user
            user_df.iloc[0],
            feature_names=feature_cols,
            matplotlib=True,
            show=False # Prevent double plotting
        )
        plt.tight_layout()
        st.pyplot(plt.gcf())
        plt.close()

        # Option 2: Show Uplift SHAP Bar Plot (More direct for uplift)
        st.write("Feature Contribution to Uplift Score")
        uplift_shap_values = treated_shap_values[0] - control_shap_values[0]
        uplift_shap_df = pd.DataFrame({
            'feature': feature_cols,
            'uplift_shap': uplift_shap_values
        }).sort_values(by='uplift_shap', key=abs, ascending=False).head(15) # Show top 15

        plt.figure(figsize=(10, 6))
        sns.barplot(x='uplift_shap', y='feature', data=uplift_shap_df, palette='viridis')
        plt.title('Top Features Contributing to Uplift Prediction')
        plt.xlabel('SHAP Value (Contribution to Uplift)')
        plt.ylabel('Feature')
        plt.tight_layout()
        st.pyplot(plt.gcf())
        plt.close()

    except Exception as e:
        st.warning(f"Could not generate SHAP explanation: {str(e)}")
        st.info("Feature importance visualization is not available, but the model predictions are still valid.")
    except Exception as e:
        st.warning(f"Could not generate SHAP explanation: {str(e)}")
        st.info("Feature importance visualization is not available, but the model predictions are still valid.")
    
    
    st.header("Recommendation")
    if uplift > 0.01:  #1% uplift threshold
        st.success("✅ TREAT this user - Significant positive uplift expected")
    elif uplift > 0:
        st.info("🟡 CONSIDER treating this user - Small positive uplift expected")
    else:
        st.error("❌ DO NOT treat this user - No positive uplift expected")

if __name__ == "__main__":
    run_simulator()
