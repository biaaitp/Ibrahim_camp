import streamlit as st
import joblib
import pandas as pd

tree = joblib.load('AI_Camp.pkl')
forest = joblib.load('RandomForest.pkl')

FastFood = pd.read_csv("https://raw.githubusercontent.com/himayatulmillah/Nutrition-Fact-for-Menu-of-McDonald/refs/heads/main/menu_mcd.csv")
features = FastFood.drop(columns=['Category', 'Item', 'Serving Size'])

st.title("🍔 What Category Is This Food?")

item_name = st.selectbox("Pick a menu item", FastFood['Item'])

if st.button("Predict Category"):
    row = features[FastFood['Item'] == item_name].iloc[[0]]
    real_answer = FastFood[FastFood['Item'] == item_name]['Category'].values[0]

    tree_guess = tree.predict(row)[0]
    forest_guess = forest.predict(row)[0]

    st.write("**Real Category:**", real_answer)
    st.write("**Decision Tree says:**", tree_guess)
    st.write("**Random Forest says:**", forest_guess)
