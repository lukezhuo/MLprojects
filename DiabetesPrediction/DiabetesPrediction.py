#pylint: skip-file
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from PIL import Image
import streamlit as st

st.write(""" 
# Diabetes Detection 
Detect if someone has diabetes using machine learning
""")

image = Image.open("Diabetes.png")
st.image(image, caption='ML Diabetes Detection', use_column_width=True)

df = pd.read_csv("diabetes.csv")
st.subheader('Data Information:')
st.dataframe(df)
st.write(df.describe())

chart = st.bar_chart(df)

X = df.iloc[:, 0:8].values
Y = df.iloc[:, -1].values

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.25, random_state = 0)

def get_user_input():
    pregnancies = st.sidebar.slider('pregnancies', 0, 20, 3)
    glucose = st.sidebar.slider('glucose', 0, 200, 100)
    blood_pressure = st.sidebar.slider('blood_pressure', 0, 125, 72)
    skin_thickness = st.sidebar.slider('skin_thickness', 0, 100, 23)
    insulin = st.sidebar.slider('insulin', 0.0, 850.0, 30.0)
    BMI = st.sidebar.slider('BMI', 0.0, 70.0, 32.0)
    diabetes_pedigree_function = st.sidebar.slider('diabetes_pedigree_function', 0.00, 2.50, 0.5)
    age = st.sidebar.slider('age', 20, 85, 30)

    #Store input in dictionary
    user_data = {
        'pregnancies': pregnancies, 
        'glucose': glucose,
        'blood_pressure': blood_pressure,
        'skin_thickness': skin_thickness,
        'insulin': insulin,
        'BMI': BMI,
        'diabetes_pedigree_function': diabetes_pedigree_function,
        'age': age
    }
    features = pd.DataFrame(user_data, index = [0])
    return features

user_input = get_user_input()

st.subheader('User Input:')
st.write(user_input)

RandomForestClassifier = RandomForestClassifier()
RandomForestClassifier.fit(X_train, Y_train)

st.subheader('Model Test Accuracy Score:')
st.write(str(accuracy_score(Y_test, RandomForestClassifier.predict(X_test))*100) + '%' )

prediction = RandomForestClassifier.predict(user_input)
st.subheader('Classification: ')
st.write(prediction)