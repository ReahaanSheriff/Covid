import numpy as np
import pandas as pd
import matplotlib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.preprocessing.sequence import TimeseriesGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM
from tensorflow.keras.models import load_model
from sklearn.metrics import mean_squared_error,mean_absolute_error,accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
from decouple import config

def lstmForecasting(filename):
    path = config('FOLDER_LOCATION')
    first = "G:/SEMESTER_6/covidForcastingProject/covid19Prediction/my_folder/images/lstmtrain.png"
    second = "G:/SEMESTER_6/covidForcastingProject/covid19Prediction/my_folder/images/lstmpredict.png"
    
    df = pd.read_csv(path+'/'+filename,parse_dates=['Date_reported'],on_bad_lines='skip')

    df = pd.DataFrame(df[df['Country']=='India'][['Date_reported','New_cases']])

    df1, df2 = train_test_split(df, test_size =.20, shuffle = False) 
    predstart = str(df2.head(1)['Date_reported'].values[0])[:10]

    df1.set_index('Date_reported',inplace=True)
    df2.set_index('Date_reported',inplace=True)

    ax = df1.plot(color='#5D3FD3', figsize=(15,6))
    try:
        first = "my_folder/images/lstmtrain.png"
        picture = ax.figure.savefig(first)
    except:
        first = "G:/SEMESTER_6/covidForcastingProject/covid19Prediction/my_folder/images/lstmtrain.png"
        picture = ax.figure.savefig(first)
    scaler = MinMaxScaler()
    scaled_df = scaler.fit_transform(df1)

    length = 50
    generator = TimeseriesGenerator(scaled_df, scaled_df, length=length, batch_size=1)

    # model = Sequential()
    # model.add(LSTM(10, activation='relu', input_shape=(length, 1), return_sequences=True)) # length = 50. number of features is 1 ('cases' column)
    # model.add(Dropout(0.4))

    # model.add(LSTM(20, activation='relu', return_sequences=True)) # length = 50. number of features is 1 ('cases' column)
    # model.add(Dropout(0.2))

    # model.add(LSTM(10, activation='relu')) 
    # model.add(Dropout(0.2))

    # model.add(Dense(1))
    # model.compile(optimizer='adam', loss='mse')

    # model.summary()

    # model.fit(generator,epochs=50)
    # model.save("/content/drive/MyDrive/ML_Trained_Models/lstm21split30epoch50")

    loaded_model = load_model('G:/SEMESTER_6/covidForcastingProject/lstm21split20')

    forecast = []
    periods = len(df2)
    first_eval_batch = scaled_df[-length:]
    current_batch = first_eval_batch.reshape((1, length, 1))

    for i in range(periods):
	    current_pred = loaded_model.predict(current_batch)[0]
	    forecast.append(current_pred) 
	    current_batch = np.append(current_batch[:,1:,:],[[current_pred]],axis=1)
    forecast = scaler.inverse_transform(forecast)
    forecast_index = pd.date_range(start=predstart,periods=periods,freq='D')
    forecast_df = pd.DataFrame(data=forecast,index=forecast_index,columns=['Forecast'])

    ax2 = df2.plot(figsize=(22,12), color='black')
    ax = forecast_df.plot(ax=ax2, color='red')
    try:
        second = "my_folder/images/lstmpredict.png"
        picture2 = ax.figure.savefig(second)
    except:
        second = "G:/SEMESTER_6/covidForcastingProject/covid19Prediction/my_folder/images/lstmpredict.png"
        picture2 = ax.figure.savefig(second)


    mae = mean_absolute_error(df2,forecast_df)
    mse = mean_squared_error(df2,forecast_df)
    rmse = np.sqrt(mse)

    return [first,second,mae,mse,rmse]
