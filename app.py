import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from pandas_datareader import data as pdr
import yfinance as yfin
from keras.models import load_model
import streamlit as st
import appdirs as _ad


yfin.pdr_override()
st.title('Stock Prediction')

user_input = st.text_input('Enter Stock Ticker', 'INFY')
user_input1 = st.date_input('Select a Start Date', key='start_date')
user_input2 = st.date_input('Select an End Date', key='end_date')

if user_input1 and user_input2:
    start_date = str(user_input1)
    end_date = str(user_input2)
    st.subheader('Data from '+ start_date +' to '+ end_date)
    df = pdr.get_data_yahoo(user_input, start_date, end_date)
    st.write(df)

#Describing data
#st.subheader('Data from '+ start_date +' to '+ end_date)
st.write(df.describe())

#visualization
st.subheader('Closing Price V/S Time Chart')
fig = plt.figure(figsize=(12,6))
plt.plot(df.Close, 'y')
st.pyplot(fig)

st.subheader('Closing Price V/S Time Chart with 100MA')
ma100 = df.Close.rolling(100).mean()
fig = plt.figure(figsize=(12,6))
plt.plot(ma100, 'r')
plt.plot(df.Close,'y' )
st.pyplot(fig)


st.subheader('Closing Price V/S Time Chart with 200MA')
ma200 = df.Close.rolling(100).mean()
fig = plt.figure(figsize=(12,6))
plt.plot(ma200, 'b')
plt.plot(df.Close, 'y')
st.pyplot(fig)


st.subheader('Closing Price V/S Time Chart with 100MA & 200MA')
ma100 = df.Close.rolling(100).mean()
ma200 = df.Close.rolling(200).mean()
fig = plt.figure(figsize=(12,6))
plt.plot(ma100, 'r')
plt.plot(ma200, 'b')
plt.plot(df.Close, 'y')
st.pyplot(fig)




#spliting data into training and testing

data_train = pd.DataFrame(df['Close'][0:int(len(df)*0.70)])
data_test = pd.DataFrame(df['Close'][int(len(df)*0.70):int(len(df))])

#normalization of data

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler(feature_range=(0,1))

#training set normalization
data_train_array = scaler.fit_transform(data_train)


#load my model

model = load_model('keras_model.h5')



#testing 
past_100_days = data_train.tail(100)
final_df = pd.concat([past_100_days, data_test], ignore_index=True)
input_data = scaler.fit_transform(final_df)

x_test =[]
y_test =[]

for i in range(100,input_data.shape[0]):
    x_test.append(input_data[i-100:i])
    y_test.append(input_data[i,0])
x_test, y_test = np.array(x_test),np.array(y_test)


#prediction

y_predict = model.predict(x_test)
y_predict = np.array(y_predict)


#scalling factor
scaler = scaler.scale_

scale_factor = 1/scaler[0]
y_predict = y_predict*scale_factor
y_test = y_test*scale_factor


#visualization
st.subheader('Prediction V/S Original')
fig2 = plt.figure(figsize=(12,6))
plt.plot(y_test, 'b' ,label='Original Price')
plt.plot(y_predict, 'r' ,label='Predicted Price')
plt.xlabel('Time')
plt.ylabel('Price')
plt.legend()
st.pyplot(fig2)