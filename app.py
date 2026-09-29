import streamlit as st
import joblib

model = joblib.load('AI_Camp.pkl')
X_test, Y_test = joblib.load('test_data.pkl')

st.title("🍔 Can the AI Guess Right?")

if st.button("Try a Real Menu Item"):
    row = X_test.sample(1)
    real_answer = Y_test.loc[row.index[0]]
    guess = model.predict(row)[0]

    st.write("Real answer:", real_answer)
    st.write("AI guessed:", guess)
    st.write("✅ Correct!" if guess == real_answer else "❌ Wrong")
