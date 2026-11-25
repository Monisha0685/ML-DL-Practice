import streamlit as st
import pandas as pd
from tensorflow.keras.models import load_model
import pickle
import os

st.title("Passenger Survival Chance in the Titanic Journey")

pclass = st.slider("Enter the Passenger class for the user : ", 1, 3)
sex = st.selectbox("Enter the passenger Gender : ", ['male', 'female'])
sibsp = st.slider("Enter the Passenger total nbumber of Siblings and Spouse : ", 1, 8)
parch = st.slider("Enter the Passenger total nbumber of Parent and Child : ", 0, 6)
fare = st.number_input("Enter the fare of the Passenger : ")
embarked = st.selectbox("Enter the Passenger station from where they started the journey : ", ['Southampton', 'Chebourg', 'Queenstown'])

user_data = pd.DataFrame([{
    'Pclass' : pclass,
    'Sex' : sex,
    'SibSp' : sibsp,
    'Parch' : parch,
    'Fare' : fare,
    'Embarked' : embarked
}])

if st.button('Data'):
    st.write(user_data)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "models")

model = load_model(MODEL_DIR, 'model.h5')

with open(MODEL_DIR, 'label_encoder.pkl', 'rb') as file:
    label = pickle.load(file)
    
with open(MODEL_DIR, 'onehot_encoder.pkl', 'rb') as file:
    onehot = pickle.load(file)
    
with open(MODEL_DIR, 'scaler.pkl', 'rb') as file:
    scaler = pickle.load(file)
    
user_data['Sex'] = label.transform(user_data['Sex'])

embarked = onehot.transform(user_data[['Embarked']])
embarked = pd.DataFrame(embarked, columns=onehot.get_feature_names_out())
user_data = pd.concat([user_data.drop(columns=['Embarked']), embarked], axis=1)

user_data[['Pclass', 'SibSp', 'Parch', 'Fare']] = scaler.transform(user_data[['Pclass', 'SibSp', 'Parch', 'Fare']])

y = model.predict(user_data)

y = y[0][0]

def Chance(y):
    if y>0.5:
        return 'The Passenger will survive the journey'
    else : 
        return 'The Passenger will not survive the journey'
    
if st.button('Predict Survival Chance'):
    st.write('Probabolity of Passenger Survival Chance', y)
    st.write(Chance(y))
