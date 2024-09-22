#!/usr/bin/env python
# coding: utf-8

# In[1]:


import warnings
warnings.filterwarnings('ignore')
import pandas as pd
data = pd.read_csv('car_purchasing.csv',encoding='ISO-8859-1')


# In[2]:


pip install xgboost


# In[3]:


data.head()


# In[4]:


data.shape


# In[5]:


data.info()


# In[6]:


data.isnull().sum()


# In[7]:


data.describe()


# In[8]:


import seaborn as sns
sns.pairplot(data)


# In[9]:


X=data.drop(['customer name','customer e-mail','country','car purchase amount'],axis=1)


# In[10]:


y = data['car purchase amount']
y


# In[11]:


from sklearn.preprocessing import MinMaxScaler



# In[12]:


sc = MinMaxScaler()
X_scaled = sc.fit_transform(X)
X_scaled


# In[13]:


sc1 = MinMaxScaler()


# In[14]:


y_reshape= y.values.reshape(-1,1)


# In[15]:


y_scaled = sc1.fit_transform(y_reshape)


# In[16]:


y_scaled


# In[17]:


from sklearn.model_selection import train_test_split


# In[18]:


X_train,X_test,y_train,y_test = train_test_split(X_scaled,y_scaled,test_size=0.20,random_state=42)


# In[19]:


data.head()


# In[20]:


from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from xgboost import XGBRegressor


# #model training

# In[22]:


lr = LinearRegression()
lr.fit(X_train,y_train)

svm = SVR()
svm.fit(X_train,y_train)

rf = RandomForestRegressor()
rf.fit(X_train,y_train)


gbr = GradientBoostingRegressor()
gbr.fit(X_train,y_train)

xg = XGBRegressor()
xg.fit(X_train,y_train)


# In[23]:


import tensorflow.keras
from keras.models import Sequential
from keras.layers import Dense
ann = Sequential()
ann.add(Dense(25, input_dim=5, activation='relu'))
ann.add(Dense(25, activation='relu'))
ann.add(Dense(1, activation='linear'))
ann.summary()


# In[24]:


ann.compile(optimizer='adam',loss='mean_squared_error')


# In[25]:


epochs_hist=ann.fit(X_train,y_train,epochs=100,batch_size=50,verbose=1,validation_split=0.2)


# prediction on test data

# In[27]:


y_pred1 = lr.predict(X_test)
y_pred2 = svm.predict(X_test)
y_pred3 = rf.predict(X_test)
y_pred4 = gbr.predict(X_test)
y_pred5 = xg.predict(X_test)
y_pred6 = ann.predict(X_test)


# In[28]:


#evaluating the algorithm
from sklearn import metrics


# In[29]:


score1 = metrics.r2_score(y_test,y_pred1)
score2 = metrics.r2_score(y_test,y_pred2)
score3 = metrics.r2_score(y_test,y_pred3)
score4 = metrics.r2_score(y_test,y_pred4)

score5 = metrics.r2_score(y_test,y_pred5)
score6 = metrics.r2_score(y_test,y_pred6)


# In[30]:


print(score1,score2,score3,score4,score5,score6)


# In[31]:


final_data = pd.DataFrame({'Models':['LR','SVR','RF','GBR','XG','ANN'],
              'R2_SCORE':[score1,score2,score3,score4,score5,score6]})


# In[32]:


final_data


# In[33]:


import seaborn as sns


# In[34]:


sns.barplot(x='Models', y='R2_SCORE', data=final_data)


# In[35]:


#save model


# In[36]:


import tensorflow.keras
from keras.models import Sequential
from keras.layers import Dense


ann = Sequential()
ann.add(Dense(25,input_dim=5,activation='relu'))
ann.add(Dense(25,activation='relu'))
ann.add(Dense(1,activation='linear'))



# In[37]:


ann.compile(optimizer='adam',loss='mean_squared_error')


# In[38]:


ann.fit(X_scaled,y_scaled,epochs=100,batch_size=50,verbose=1)


# In[39]:


import joblib

 


# In[40]:


joblib.dump(ann,'car_model')
 


# In[41]:


model = joblib.load('car_model')


# In[42]:


import matplotlib.pyplot as plt


# In[43]:


plt.plot(epochs_hist.history['loss'])
plt.plot(epochs_hist.history['val_loss'])
plt.title('model lost progess during training')
plt.ylabel('Traning and validation loss')
plt.xlabel('epochs number')
plt.legend(['training loss','validation loss'])


# In[44]:


y_pred=model.predict(X_test)


# In[45]:


y_pred


# In[46]:


import numpy as np


# In[47]:


data.head(1)


# In[48]:


X_test1=sc.transform(np.array([[0,42,62812.09301,11609.38091,238961.2505]]))


# In[49]:


X_test1


# In[50]:


pred=ann.predict(X_test1)


# In[51]:


p=sc1.inverse_transform(pred)


# In[52]:


p


# In[117]:


#pip3 install ann_visualizer


# In[107]:


#pip install graphviz


# In[119]:


#from ann_visualizer.visualize import ann_viz;


# In[121]:


#ann_viz(ann,title="ann")


# In[ ]:


#GUI


# In[125]:


import numpy as np
from tkinter import *
from sklearn.preprocessing  import StandardScaler
import joblib


# In[127]:


def show_entry_fields():
    p1=float(e1.get())
    p2=float(e2.get())
    p3=float(e3.get())
    p4=float(e4.get())
    p5=float(e5.get())
    
    model = joblib.load('car_model')
    result=model.predict(sc.transform(np.array([[p1,p2,p3,p4,p5]])))
    Label(master, text="Car Purchase amount").grid(row=8)
    Label(master, text=sc1.inverse_transform(result)).grid(row=10)
    print("Car Purchase amount", sc1.inverse_transform(result)[0][0])
    
master = Tk()
master.title("Car Purchase Amount Predictions Using Machine Learning")


label = Label(master, text = "Car Purchase Amount Predictions Using ML"
                          , bg = "black", fg = "white"). \
                               grid(row=0,columnspan=2)


Label(master, text="Gender").grid(row=1)
Label(master, text="Age").grid(row=2)
Label(master, text="Annual Salary").grid(row=3)
Label(master, text="Credit Card Debt").grid(row=4)
Label(master, text="Net Worth").grid(row=5)


e1 = Entry(master)
e2 = Entry(master)
e3 = Entry(master)
e4 = Entry(master)
e5 = Entry(master)

e1.grid(row=1, column=1)
e2.grid(row=2, column=1)
e3.grid(row=3, column=1)
e4.grid(row=4, column=1)
e5.grid(row=5, column=1)


Button(master, text='Predict', command=show_entry_fields).grid()

mainloop()

