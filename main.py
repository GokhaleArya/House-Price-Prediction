import streamlit as st
import pickle
import numpy as np

with open('./src/models/catboost_optuna.pkl', 'rb') as file:
    model = pickle.load(file)

st.markdown(
    """
    <h1 style='text-align: center;'>House Price Predictor</h1>
    """,
    unsafe_allow_html=True
)

st.sidebar.header("Enter the House Features")

MSSubClass_options = {
    20: "1-STORY 1946 & NEWER ALL STYLES",
    30: "1-STORY 1945 & OLDER",
    40: "1-STORY W/FINISHED ATTIC ALL AGES",
    45: "1-1/2 STORY - UNFINISHED ALL AGES",
    50: "1-1/2 STORY FINISHED ALL AGES",
    60: "2-STORY 1946 & NEWER",
    70: "2-STORY 1945 & OLDER",
    75: "2-1/2 STORY ALL AGES",
    80: "SPLIT OR MULTI-LEVEL",
    85: "SPLIT FOYER",
    90: "DUPLEX - ALL STYLES AND AGES",
    120: "1-STORY PUD - 1946 & NEWER",
    150: "1-1/2 STORY PUD - ALL AGES",
    160: "2-STORY PUD - 1946 & NEWER",
    180: "PUD - MULTILEVEL - INCL SPLIT LEV/FOYER",
    190: "2 FAMILY CONVERSION - ALL STYLES AND AGES"
}

MSSubClass = st.sidebar.selectbox(
    "MSSubClass",
    options=list(MSSubClass_options.keys()),  
    format_func=lambda x: MSSubClass_options[x]  
)

BldgType_options = {
    0: "1Fam",
    1: "2FmCon",
    2: "Duplex",
    3: "TwnhsE",
    4: "Twnhs"
}

BldgType = st.sidebar.selectbox(
    "BldgType",
    options=list(BldgType_options.keys()),  
    format_func=lambda x: BldgType_options[x]
)

PoolQC_options = {
    0: "Ex",  # Excellent
    1: "Gd",  # Good
    2: "TA",  # Average/Typical
    3: "Fa",  # Fair
    4: "NA"   # No Pool
}

PoolQC = st.sidebar.selectbox(
    "PoolQC",
    options=list(PoolQC_options.keys()),  
    format_func=lambda x: PoolQC_options[x]  
)

OverallQual = st.sidebar.slider("OverallQual (1-10)", 1, 10, 5)
GrLivArea = st.sidebar.number_input("GrLivArea (sq ft)", min_value=400, max_value=6000, value=1500)
GarageCars = st.sidebar.slider("GarageCars", 0, 4, 1)
YearBuilt = st.sidebar.number_input("YearBuilt", min_value=1872, max_value=2010, value=1900)
TotalBsmtSF = st.sidebar.number_input("TotalBsmtSF (sq ft)", min_value=0, max_value=6000, value=1000)
Fireplaces = st.sidebar.slider("Fireplaces", 0, 3, 1)
LotArea = st.sidebar.number_input("LotArea (sq ft)", min_value=1000, max_value=200000, value=7000)

if st.sidebar.button("Predict House Price"):

    GrLivArea_ = (GrLivArea - 1488.55821918 )/ np.sqrt(280607.1438708)
    TotalBsmtSF_ = (TotalBsmtSF - 1043.85273973)/np.sqrt(183964.33790345)
    LotArea_ = (LotArea - 9769.50342466)/np.sqrt(29927006.05820744)

    input_features = [[MSSubClass, OverallQual, GrLivArea_, GarageCars, BldgType, YearBuilt,
                       TotalBsmtSF_, Fireplaces, LotArea_, PoolQC]]
    
    output_price = model.predict(input_features)
    st.markdown(f"""
        <h4>The Price of the house based on features below is : </h4><h2><strong>${output_price[0]:,.2f} \u00B1 18000 (Mean Absolute)</strong></h2>
    """, unsafe_allow_html=True)
    
    st.write('')
    st.write('')
    st.write('')

    st.markdown('''

    #### **INPUT FEATURES**

    ''')    

    st.write(f"Selected MSSubClass : {MSSubClass_options[MSSubClass]}")
    st.write(f"Selected BldgType : {BldgType_options[BldgType]}")
    st.write(f"Selected PoolQC : {PoolQC_options[PoolQC]}")
    st.write(f"OverallQual: {OverallQual}")
    st.write(f"GrLivArea: {GrLivArea} sq ft")
    st.write(f"GarageCars: {GarageCars}")
    st.write(f"YearBuilt: {YearBuilt}")
    st.write(f"TotalBsmtSF: {TotalBsmtSF} sq ft")
    st.write(f"Fireplaces: {Fireplaces}")
    st.write(f"LotArea: {LotArea} sq ft")




