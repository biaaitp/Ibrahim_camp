%%writefile app.py
import streamlit as st
import joblib

tree = joblib.load('AI_Camp.pkl')
forest = joblib.load('RandomForest.pkl')
X_test, Y_test = joblib.load('test_data.pkl')

st.title("🍔 McDonald's Menu AI — Model Showdown")
st.write("Comparing two real AI models trained on real McDonald's menu data.")

tree_accuracy = tree.score(X_test, Y_test)
forest_accuracy = forest.score(X_test, Y_test)

col1, col2 = st.columns(2)
col1.metric("Decision Tree Accuracy", f"{tree_accuracy:.0%}")
col2.metric("Random Forest Accuracy", f"{forest_accuracy:.0%}")

st.bar_chart({
    "Decision Tree": tree_accuracy,
    "Random Forest": forest_accuracy
})

st.write("---")

if st.button("🎲 Try a Real Menu Item"):
    row = X_test.sample(1)
    real_answer = Y_test.loc[row.index[0]]
    tree_guess = tree.predict(row)[0]
    forest_guess = forest.predict(row)[0]

    st.write("**Real answer:**", real_answer)
    st.write("**Decision Tree guessed:**", tree_guess, "✅" if tree_guess == real_answer else "❌")
    st.write("**Random Forest guessed:**", forest_guess, "✅" if forest_guess == real_answer else "❌")
