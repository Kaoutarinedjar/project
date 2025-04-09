{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 13,
   "id": "a0f3b427-46c8-4038-9021-733109b1d07b",
   "metadata": {},
   "outputs": [
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "2025-04-09 02:44:49.630 \n",
      "  \u001b[33m\u001b[1mWarning:\u001b[0m to view this Streamlit app on a browser, run it with the following\n",
      "  command:\n",
      "\n",
      "    streamlit run C:\\ProgramData\\anaconda3\\Lib\\site-packages\\ipykernel_launcher.py [ARGUMENTS]\n"
     ]
    }
   ],
   "source": [
    "import streamlit as st\n",
    "import joblib\n",
    "import numpy as np\n",
    "\n",
    "# Charger le modèle\n",
    "model = joblib.load(\"Mobile_Prices.pkl\")\n",
    "\n",
    "# Liste de toutes les features dans l’ordre attendu par le modèle\n",
    "feature_names = [\n",
    "    'battery_power', 'blue', 'clock_speed', 'dual_sim', 'fc',\n",
    "    'four_g', 'int_memory', 'm_dep', 'mobile_wt', 'n_cores',\n",
    "    'pc', 'px_height', 'px_width', 'ram', 'sc_h',\n",
    "    'sc_w', 'talk_time', 'three_g', 'touch_screen', 'wifi'\n",
    "]\n",
    "\n",
    "# Dictionnaire pour stocker les entrées utilisateur\n",
    "input_data = []\n",
    "\n",
    "st.title(\"Prédicteur de prix de téléphone mobile\")\n",
    "\n",
    "for feature in feature_names:\n",
    "    if feature in ['blue', 'dual_sim', 'four_g', 'three_g', 'touch_screen', 'wifi']:\n",
    "        val = st.selectbox(f\"{feature}\", [0, 1])\n",
    "    else:\n",
    "        val = st.number_input(f\"{feature}\", value=0.0)\n",
    "    input_data.append(val)\n",
    "\n",
    "if st.button(\"Prédire le prix\"):\n",
    "    features = np.array([input_data])\n",
    "    prediction = model.predict(features)\n",
    "    st.success(f\"Classe de prix prédite : {prediction[0]}\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "id": "d55181cf-c355-462d-8667-ecd7c918a82d",
   "metadata": {},
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "id": "ee46963a-e955-4548-9311-765afc6e0163",
   "metadata": {},
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3 (ipykernel)",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.12.4"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
