import streamlit as st
import pandas as pd
import pickle

model = pickle.load(open("model.pkl", "rb"))
columns = pickle.load(open("columns.pkl", "rb"))

st.title("Student Performance Prediction")

gender = st.selectbox("Gender", ["male", "female"])
race = st.selectbox("Race", ["group A","group B","group C","group D","group E"])
parent = st.selectbox("Parent Education", [
"some high school","high school","some college",
"associate's degree","bachelor's degree","master's degree"
])
lunch = st.selectbox("Lunch", ["standard","free/reduced"])
test = st.selectbox("Test Preparation", ["none","completed"])

reading = st.slider("Reading Score",0,100,50)
writing = st.slider("Writing Score",0,100,50)

if st.button("Predict"):

    input_dict = {
        "gender": gender,
        "race/ethnicity": race,
        "parental level of education": parent,
        "lunch": lunch,
        "test preparation course": test,
        "reading score": reading,
        "writing score": writing
    }

    input_df = pd.DataFrame([input_dict])

    input_df = pd.get_dummies(input_df)

    input_df = input_df.reindex(columns=columns, fill_value=0)

    prediction = model.predict(input_df)

    st.success(f"Predicted Math Score: {prediction[0]:.2f}")