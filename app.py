#!/usr/bin/env python
# coding: utf-8

# In[2]:


import streamlit as st
import joblib
import numpy as np
import os

# Charger le modèle
model = joblib.load("Mobile_Prices.pkl")

# Liste des features
feature_names = [
    'battery_power', 'blue', 'clock_speed', 'dual_sim', 'fc',
    'four_g', 'int_memory', 'm_dep', 'mobile_wt', 'n_cores',
    'pc', 'px_height', 'px_width', 'ram', 'sc_h',
    'sc_w', 'talk_time', 'three_g', 'touch_screen', 'wifi'
]

# Titre de l'app
st.title("Prédicteur de prix de téléphone mobile")

# Collecte des données utilisateur
input_data = []

for feature in feature_names:
    if feature in ['blue', 'dual_sim', 'four_g', 'three_g', 'touch_screen', 'wifi']:
        val = st.selectbox(f"{feature}", [0, 1])
        input_data.append(int(val))
    else:
        val = st.number_input(f"{feature}", min_value=0.0, value=0.0, step=0.1, format="%.2f")
        input_data.append(float(val))

# Prédiction
if st.button("Prédire le prix"):
    features = np.array([input_data])
    prediction = model.predict(features)
    st.success(f"Classe de prix prédite : {prediction[0]}")


# In[4]:





# In[ ]:





# In[ ]:




