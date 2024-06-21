#!/usr/bin/env python
# coding: utf-8

# In[1]:


from sklearn.preprocessing import StandardScaler
import pandas as pd
import streamlit as st
import pickle


# In[9]:


with open('desktop/Project/Red_Wine_Quality_Determination/red_wine_quality_determination.pkl', 'rb')as file:
    model = pickle.load(file)


# In[10]:


st.title('Red Wine Quality Determination App')
st.markdown(
    'Enter the fields below to get an estimate of the quality of your red wine...')


# In[11]:


fixed_acidity = st.number_input('fixed acidity of wine', format='%2f')

volatile_acidity = st.number_input('volatile acidity of wine', format='%2f')

citric_acid = st.number_input('citric acid of wine', format='%2f')

residual_sugar = st.number_input('residual sugar of wine', format='%2f')

chlorides = st.number_input('chlorides of wine', format='%3f')

free_sulphur_dioxide = st.number_input(
    'free sulphur dioxide of wine', format='%2f')

total_sulphur_dioxide = st.number_input(
    'total sulphur dioxide of wine', format='%2f')

density = st.number_input('density of wine', format='%4f')

pH = st.number_input('pH of wine', format='%2f')

sulphates = st.number_input('sulphates of wine', format='%2f')

alcohol = st.number_input('alcohol of wine', format='%2f')

predict = st.button('Predict')


# In[12]:


if predict:
    columns = ['fixed acidity', 'volatile acidity', 'citric acid', 'residual sugar', 'chlorides', 'free sulphur dioxide', 'total sulphur dioxide', 'density', 'pH',
               'sulphates', 'alcohol']
    data = [[fixed_acidity, volatile_acidity, citric_acid, residual_sugar, chlorides, free_sulphur_dioxide, total_sulphur_dioxide, density,
             pH, sulphates, alcohol]]
    X = pd.DataFrame(data, columns)
    ss = StandardScaler()
    scaled_X = ss.fit_transform(X)
    st.write('Is the Red Wine of Good Quality: ')
    st.write(model.predict(scaled_X)[0])


# In[ ]:
