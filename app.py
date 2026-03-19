
import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# --- Enforce Light UI Styling ---
st.set_page_config(page_title="Symptom MediCare", layout="centered")
st.markdown("""
    <style>
        .stApp {
            background-color: #ffffff;
            color: #111;
        }

        h1, h2, h3, .stMarkdown p {
            color: #111 !important;
        }

        .stButton > button {
            background-color: #1565c0 !important;
            color: white !important;
            border: none;
            border-radius: 8px;
            padding: 10px 18px;
            font-size: 16px;
        }

        .stButton > button:hover {
            background-color: #0d47a1 !important;
        }

        div[data-baseweb="select"] > div {
            background-color: white !important;
            color: black !important;
        }
    </style>
""", unsafe_allow_html=True)

# --- Title & Intro ---
st.title("Symptom MediCare 🩺")
st.markdown("""
Welcome to **Symptom MediCare**!  
This tool predicts the likelihood of **Malaria**, **Typhoid**, or **HIV/AIDS** based on symptoms you select.
""")

# --- Symptom Options ---
symptom_option_mapping = {
    'Fever': ['High', 'Medium', 'Low'],
    'Fatigue': ['Very High', 'High', 'Medium', 'Low'],
    'Headache': ['Very High (Heaviness)', 'High', 'Medium', 'Low'],
    'Vomiting': ['Yes', 'No'],
    'Skin Rash': ['High', 'Medium', 'Mild', 'None', 'Rose spots'],
    'Muscle Joint Pain': ['Yes', 'No', 'Medium'],
    'Weight Loss': ['Severe', 'Moderate', 'Mild'],
    'Diarrhea': ['Yes', 'No'],
    'Night Sweats': ['Yes', 'No'],
    'Lymph Node Swelling': ['High', 'Yes', 'No']
}

# --- Symptom Dataset ---
data = {
    'Disease': ['Malaria', 'Malaria', 'Malaria',
                'Typhoid', 'Typhoid', 'Typhoid',
                'HIV/AIDS', 'HIV/AIDS', 'HIV/AIDS'],
    'Fever': ['High', 'Medium', 'High',
              'High', 'High', 'Medium',
              'Medium', 'Low', 'Low'],
    'Fatigue': ['Very High', 'High', 'High',
                'High', 'Medium', 'Low',
                'Very High', 'High', 'High'],
    'Headache': ['High', 'Medium', 'High',
                 'Very High (Heaviness)', 'High', 'Medium',
                 'Low', 'Medium', 'Medium'],
    'Vomiting': ['Yes', 'Yes', 'Yes',
                 'Yes', 'Yes', 'No',
                 'Yes', 'Yes', 'Yes'],
    'Skin Rash': ['Mild', 'Mild', 'None',
                  'Rose spots', 'Mild', 'None',
                  'High', 'High', 'Medium'],
    'Muscle Joint Pain': ['Yes', 'No', 'Medium',
                          'No', 'Yes', 'Medium',
                          'Yes', 'Yes', 'Yes'],
    'Weight Loss': ['Moderate', 'Mild', 'Severe',
                    'Mild', 'Mild', 'Moderate',
                    'Severe', 'Severe', 'Severe'],
    'Diarrhea': ['No', 'Yes', 'No',
                 'Yes', 'No', 'Yes',
                 'Yes', 'Yes', 'Yes'],
    'Night Sweats': ['Yes', 'Yes', 'No',
                     'No', 'No', 'No',
                     'Yes', 'Yes', 'Yes'],
    'Lymph Node Swelling': ['No', 'No', 'No',
                             'No', 'No', 'No',
                             'Yes', 'Yes', 'High']
}
df = pd.DataFrame(data)

# --- Symptom Form ---
user_symptoms = {}
with st.form("symptom_form"):
    st.markdown("### 📝 Select the options that best describe your symptoms:")

    for symptom, options in symptom_option_mapping.items():
        user_symptoms[symptom] = st.selectbox(f"**Select {symptom}**", options, key=symptom)

    submitted = st.form_submit_button("🧪 Predict Disease")

# --- Naive Bayes Classifier ---
def predict_disease(df, user_symptoms):
    disease_probs = {}
    total_count = len(df)
    diseases = df['Disease'].unique()

    for disease in diseases:
        sub_df = df[df['Disease'] == disease]
        prior = len(sub_df) / total_count
        likelihood = 1.0

        for symptom, value in user_symptoms.items():
            match_count = len(sub_df[sub_df[symptom] == value])
            symptom_prob = (match_count + 1) / (len(sub_df) + len(df[symptom].unique()))
            likelihood *= symptom_prob

        disease_probs[disease] = prior * likelihood

    total_prob = sum(disease_probs.values())
    if total_prob == 0:
        return "No Match Found", {}

    for disease in disease_probs:
        disease_probs[disease] = (disease_probs[disease] / total_prob) * 100

    predicted = max(disease_probs, key=disease_probs.get)
    return predicted, disease_probs

# --- Display Results ---
if submitted:
    prediction, probs = predict_disease(df, user_symptoms)
    confidence = probs.get(prediction, 0)

    st.success(f"🎯 Based on your symptoms, the most likely disease is: **{prediction}**")
    st.info(f"🧪 Prediction Confidence: **{confidence:.2f}%**")

    # --- Chart ---
    if probs:
        st.markdown("### 📊 Symptom MediCare Prediction Probability Chart")
        fig, ax = plt.subplots()
        diseases = list(probs.keys())
        values = list(probs.values())
        sns.barplot(x=diseases, y=values, palette='coolwarm', ax=ax)

        for i, (disease, prob) in enumerate(zip(diseases, values)):
            ax.text(i, prob + 1, f"{prob:.1f}%", ha='center', va='bottom',
                    fontsize=10, color='black', fontweight='bold')

        ax.set_ylabel("Probability (%)")
        ax.set_title("Symptom MediCare Disease Prediction")
        ax.set_ylim(0, max(values) + 10)
        st.pyplot(fig)

# --- Sidebar Info ---
st.sidebar.header("About")
st.sidebar.info("""
**Symptom MediCare**  
Symptom MediCare is a health-simulated demo project aims to tackle critical health challenges by providing a smarter way to diagnose diseases based on physical symptoms. 
It seeks to reduce misdiagnosis and the inappropriate use of antimalarial and antibiotic drugs by the public.
By guiding healthcare workers toward more accurate assessmentsin the earlier detection and diagnosis of diseases.
Ensuring patients receive the right treatment at the right time.

**Created by:** Edidiong Moses  
**Initiated by:** 3MTT Nigeria  
**Built with:** Streamlit + Naive Bayes
""")
