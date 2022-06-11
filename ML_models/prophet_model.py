import pandas as pd
from fbprophet.plot import plot_plotly, plot_components_plotly
from fbprophet import Prophet
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from statsmodels.tools.eval_measures import rmse
import math
from decouple import config

def prophetForecasting(filename):
    firstpro = "G:/SEMESTER_6/covidForcastingProject/covid19Prediction/my_folder/images/protrain.png"
    secondpro = "G:/SEMESTER_6/covidForcastingProject/covid19Prediction/my_folder/images/propredict.png"
    path = config('FOLDER_LOCATION')
    df = pd.read_csv(path+'/'+filename,parse_dates=['Date_reported'],on_bad_lines='skip')
    df = pd.DataFrame(df[df['Country']=='India'][['Date_reported','New_cases']])
    df.columns = ['ds','y']
    df['ds'] = pd.to_datetime(df['ds'], dayfirst = True)
    train, test = train_test_split(df, test_size=0.2, shuffle = False)
    ax = train.plot(x='ds',y='y',figsize=(18,6))
    try:
        firstpro = "my_folder/images/protrain.png"
        picture1 = ax.figure.savefig(firstpro)
    except:
        firstpro = "G:/SEMESTER_6/covidForcastingProject/covid19Prediction/my_folder/images/protrain.png"
        picture1 = ax.figure.savefig(firstpro)
    # fig.write_image("images/fig1.png")
    # m.plot_components(forecast).savefig('1.png')
    m = Prophet()
    m.fit(train)
    future = m.make_future_dataframe(periods=len(test)) #MS for monthly, H for hourly
    forecast = m.predict(future)
    forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']]
    ax = plot_plotly(m ,forecast)
    #picture2 = ax.figure.savefig(secondpro)
    try:
        secondpro = "my_folder/images/propredict.png"
        picture2 = ax.write_image(secondpro)
    except:
        secondpro = "G:/SEMESTER_6/covidForcastingProject/covid19Prediction/my_folder/images/propredict.png"
        picture2 = ax.write_image(secondpro)
    metric_df = forecast.set_index('ds')[['yhat']].join(df.set_index('ds').y).reset_index()
    metric_df.dropna(inplace=True)
    r2_score(metric_df.y, metric_df.yhat)
    mse = mean_squared_error(metric_df.y, metric_df.yhat)
    mae = mean_absolute_error(metric_df.y, metric_df.yhat)
    rmse = math.sqrt(mean_squared_error(metric_df.y, metric_df.yhat))
    return [firstpro,secondpro,mae,mse,rmse]
