import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import datetime as dt
from sklearn.metrics import mean_squared_error,r2_score, mean_absolute_error
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split
import pmdarima as pm
from decouple import config

def arimaForecasting(filename):
    path = config('FOLDER_LOCATION')
    first = "G:/SEMESTER_6/covidForcastingProject/covid19Prediction/my_folder/images/arpredict.png"

    
    df = pd.read_csv(path+'/'+filename,parse_dates=['Date_reported'],on_bad_lines='skip')

    df = pd.DataFrame(df[df['Country']=='India'][['Date_reported','New_cases']])
    df.set_index('Date_reported',inplace=True)
    df1, df2 = train_test_split(df, test_size =.10, shuffle = False)
    y_pred = df2.copy()

    model_arima= pm.auto_arima(df1["New_cases"],trace=True, error_action='ignore', start_p=1,start_q=1,max_p=3,max_q=3,
                   suppress_warnings=True,stepwise=False,seasonal=False)
    model_arima.fit(df1["New_cases"])

    prediction_arima=model_arima.predict(len(df2))
    y_pred["ARIMA Model Prediction"]=prediction_arima

    model_scores=[]
    model_scores.append(np.sqrt(mean_squared_error(y_pred["New_cases"],y_pred["ARIMA Model Prediction"])))
    rmse = np.sqrt(mean_squared_error(y_pred["New_cases"],y_pred["ARIMA Model Prediction"]))
    mse = mean_squared_error(y_pred["New_cases"],y_pred["ARIMA Model Prediction"])
    mae = mean_absolute_error(y_pred["New_cases"],y_pred["ARIMA Model Prediction"])
    fig=go.Figure()
    fig.add_trace(go.Scatter(x=df1.index, y=df1["New_cases"],
                        mode='lines+markers',name="Train Data for Confirmed Cases"))
    fig.add_trace(go.Scatter(x=df2.index, y=df2["New_cases"],
                        mode='lines+markers',name="Validation Data for Confirmed Cases",))
    fig.add_trace(go.Scatter(x=df2.index, y=y_pred["ARIMA Model Prediction"],
                        mode='lines+markers',name="Prediction for Confirmed Cases",))
    fig.update_layout(title="Confirmed Cases ARIMA Model Prediction",
                    xaxis_title="Date",yaxis_title="Confirmed Cases",legend=dict(x=0,y=1,traceorder="normal"))
    #fig.show()
    try:
        first = "my_folder/images/arpredict.png"
        fig.write_image(first) # save as test.png
    except:
        first = "G:/SEMESTER_6/covidForcastingProject/covid19Prediction/my_folder/images/arpredict.png"
        fig.write_image(first) # save as test.png
    return [first, rmse, mse, mae]