import streamlit as st
import joblib
import pandas as pd

tree = joblib.load('AI_Camp.pkl')
forest = joblib.load('RandomForest.pkl')

FastFood = pd.read_csv("https://raw.githubusercontent.com/himayatulmillah/Nutrition-Fact-for-Menu-of-McDonald/refs/heads/main/menu_mcd.csv")
features = FastFood.drop(columns=['Category', 'Item', 'Serving Size'])

st.title("🍔 Can the AI Guess the Category?")
st.write("Pick any real McDonald's item and watch two AI brains guess live!")

item_name = st.selectbox("🍟 Pick a menu item", FastFood['Item'])

if st.button("🔮 Predict Category"):
    row = features[FastFood['Item'] == item_name].iloc[[0]]
    real_answer = FastFood[FastFood['Item'] == item_name]['Category'].values[0]

    tree_guess = tree.predict(row)[0]
    forest_guess = forest.predict(row)[0]

    st.write("---")
    st.subheader(f"✅ Real Answer: **{real_answer}**")

    col1, col2 = st.columns(2)
    with col1:
        st.write("🌳 **Decision Tree guessed:**")
        if tree_guess == real_answer:
            st.success(f"{tree_guess} ✅")
        else:
            st.error(f"{tree_guess} ❌")

    with col2:
        st.write("🌲 **Random Forest guessed:**")
        if forest_guess == real_answer:
            st.success(f"{forest_guess} ✅")
        else:
            st.error(f"{forest_guess} ❌")

    if tree_guess == real_answer and forest_guess == real_answer:
        st.balloons()
