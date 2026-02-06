# importing the necessary libraries
import streamlit as st
import pandas as pd
import numpy as np
import xgboost as xgb

# setting up the streamlit page
st.set_page_config(page_title="Nanofluid HTC Predictor", page_icon="🌡️", layout="wide")

# setting the custom CSS for button and card elements
st.markdown("""
    <style>
    /* main button */
    div.stButton > button:first-child {
        background-color: #FF4B4B;
        color: white;
        height: 2.2em;
        width: 100%;
        border-radius: 10px;
        border: none;
        font-weight: bold;
        font-size: 22px;
        transition: 0.3s;
        box-shadow: 0px 4px 10px rgba(0,0,0,0.1);
    }
    div.stButton > button:hover {
        background-color: #D32F2F;
        color: white;
        box-shadow: 0px 6px 15px rgba(0,0,0,0.2);
    }
    /* results card */
    .result-card {
        background-color: #f0f2f6;
        border-left: 10px solid #FF4B4B;
        padding: 20px;
        border-radius: 10px;
        margin-bottom: 20px;
    }
    .metric-label {
        color: #555;
        font-size: 14px;
        font-weight: bold;
    }
    .metric-value {
        color: #FF4B4B;
        font-size: 32px;
        font-weight: bold;
    }
    .config-note {
        font-size: 12px;
        color: #666;
        font-style: italic;
    }
    </style>
    """, unsafe_allow_html=True)

# setting the header section
st.title("🌡️ Multiphase Interface: Nanofluid Heat Transfer Prediction System")
st.markdown("""
The model predicts the **Heat Transfer Coefficient (HTC)** of nanofluids based on the interaction between the dispersed solid phase and the continuous liquid phase.
            As the volume fraction increases, the interfacial area expands, typically leading to enhanced energy transport.
""")

# setting the results container
result_container = st.container()

st.markdown("---")

# setting the sidebar for fluid and particle parameters input to the model
st.sidebar.header("Fluid & Particle Parameters")
nano_type = st.sidebar.selectbox("Nanoparticle Type", ["Al2O3", "CuO", "SiO2", "TiO2", "ZnO"])
base_fluid = st.sidebar.selectbox("Base Fluid", ["Water", "Ethylene Glycol", "Oil", "Propylene Glycol"])
vol_fraction = st.sidebar.slider("Volume Fraction (%)", 0.1, 5.0, 1.0)
temp = st.sidebar.number_input("Temperature (°C)", value=40.0)
velocity = st.sidebar.number_input("Flow Velocity (m/s)", value=1.5)

# writing information about the system geometry
st.info("ℹ️ **System Configuration:** This model assumes a pipe flow geometry with a fixed characteristic length (Diameter) of **0.01 m (10 mm)**.")

# settting the section for inputting thermophysical properties
st.subheader("🧪 Thermophysical Properties")
col1, col2, col3, col4 = st.columns(4)

with col1:
    density = st.number_input("Density (kg/m³)", value=1000.0)
with col2:
    viscosity = st.number_input("Viscosity (Pa·s)", value=0.001, format="%.4f")
with col3:
    cond = st.number_input("Thermal Conductivity (W/mK)", value=0.6)
with col4:
    sp_heat = st.number_input("Specific Heat (J/kgK)", value=4180.0)

st.markdown("###") 

# setting the prediction button
predict_btn = st.button("Run HTC Prediction")

# adding the prediction logic
if predict_btn:
    # loading the saved model
    model = xgb.XGBRegressor()
    model.load_model("./model_assets/htc_pred_xgb_model.json")

    # calculating the Re and Pr
    D = 0.01  # Fixed diameter as used in the notebook
    reynolds = (density * velocity * D) / viscosity
    prandtl = (sp_heat * viscosity) / cond

    # setting the input columns
    expected_columns = [
        'Volume_Fraction (%)', 'Temperature (°C)', 'Flow_Velocity (m/s)', 
        'Thermal_Conductivity (W/mK)', 'Specific_Heat_Capacity (J/kgK)', 
        'Density (kg/m³)', 'Viscosity (Pa·s)', 'Reynolds_Number', 'Prandtl_Number', 
        'Nanoparticle_Type_CuO', 'Nanoparticle_Type_SiO2', 'Nanoparticle_Type_TiO2', 
        'Nanoparticle_Type_ZnO', 'Base_Fluid_Oil', 'Base_Fluid_Propylene Glycol', 
        'Base_Fluid_Water'
    ]

    input_data = {col: 0.0 for col in expected_columns}
    input_data.update({
        'Volume_Fraction (%)': vol_fraction, 'Temperature (°C)': temp,
        'Flow_Velocity (m/s)': velocity, 'Thermal_Conductivity (W/mK)': cond,
        'Specific_Heat_Capacity (J/kgK)': sp_heat, 'Density (kg/m³)': density,
        'Viscosity (Pa·s)': viscosity, 'Reynolds_Number': reynolds, 'Prandtl_Number': prandtl
    })

    if f'Nanoparticle_Type_{nano_type}' in input_data:
        input_data[f'Nanoparticle_Type_{nano_type}'] = 1
    if f'Base_Fluid_{base_fluid}' in input_data:
        input_data[f'Base_Fluid_{base_fluid}'] = 1

    # fetching the prediction
    input_df = pd.DataFrame([input_data])[expected_columns]
    prediction = model.predict(input_df)[0]

    # displaying the results
    with result_container:
        st.markdown(f"""
            <div class="result-card">
                <h3 style='margin-top:0;'>Model Analysis Complete ✅</h3>
                <div style="display: flex; justify-content: space-between; flex-wrap: wrap;">
                    <div style="margin-right: 20px;">
                        <p class="metric-label">PREDICTED HTC</p>
                        <p class="metric-value">{prediction:.2f} W/m²K</p>
                    </div>
                    <div style="margin-right: 20px;">
                        <p class="metric-label">REYNOLDS NUMBER (Re)</p>
                        <p class="metric-value" style="color:#2E7D32;">{reynolds:.0f}</p>
                    </div>
                    <div style="margin-right: 20px;">
                        <p class="metric-label">PRANDTL NUMBER (Pr)</p>
                        <p class="metric-value" style="color:#1565C0;">{prandtl:.2f}</p>
                    </div>
                    <div>
                        <p class="metric-label">CHAR. LENGTH (D)</p>
                        <p class="metric-value" style="color:#7B1FA2;">{D} m</p>
                    </div>
                </div>
            </div>
        """, unsafe_allow_html=True)
        # addinng the insights section
        st.info(f"**Insight:** At a volume fraction of {vol_fraction}%, the model suggests the dispersed phase is effectively {'enhancing' if prediction > 4000 else 'maintaining'} the thermal boundary layer stability.")
