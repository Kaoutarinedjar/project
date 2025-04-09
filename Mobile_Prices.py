#!/usr/bin/env python
# coding: utf-8

# In[28]:


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt


# In[29]:


df = pd.read_csv("mobile_prices.csv")
df.head()


# In[30]:


print("\nValeurs manquantes :\n", df.isnull().sum())


# In[31]:


X = df.drop("price_range", axis=1)
y = df["price_range"]


# In[32]:


scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)


# In[33]:


X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)


# In[34]:


log_regression = LogisticRegression(multi_class='multinomial', solver='lbfgs', max_iter=1000)
log_regression.fit(X_train, y_train)


# In[35]:


y_pred = log_reg.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(" Accuracy :",accuracy)



# In[36]:


cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=[0,1,2,3], yticklabels=[0,1,2,3])
plt.xlabel('Prédiction')
plt.ylabel('Vraie valeur')
plt.title('Matrice de confusion - Régression Logistique')
plt.show()


# In[38]:


import joblib
joblib.dump(log_regression,'Mobile_Prices.pkl')


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




