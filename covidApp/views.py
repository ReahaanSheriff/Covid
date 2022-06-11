from django.shortcuts import render,redirect
from django.conf import settings
from django.core.files.storage import FileSystemStorage
import os
from decouple import config
import numpy as np
import pandas as pd
import matplotlib
import plotly.express as px
import matplotlib.pyplot as plt
import seaborn as sns
from ML_models.LSTM_model import lstmForecasting
from ML_models.prophet_model import prophetForecasting
from ML_models.arima_model import arimaForecasting
from ML_models.lockdown import lock
import dataframe_image as dfi
# Create your views here.

selected = ''
mse_vals = []
mae_vals = []
rmse_vals = []

def home(request):
    folder='my_folder/dataset/'
    if request.method == 'POST' and request.FILES['csvFile']:

        fs = FileSystemStorage(location=folder) #defaults to   MEDIA_ROOT
        path = config('FOLDER_LOCATION')
        dir_list = os.listdir(path)
        myfile = request.FILES.getlist('csvFile')
        for i in myfile:
            print(i)
            nam = i.name
            if(nam  in dir_list):
                os.remove(path+'/'+nam)
            t = nam.find('.')
            if(nam[t+1:] == 'csv'):
                filename = fs.save(i.name, i)
                file_url = fs.url(filename)
                msg = 'File uploaded successfully'
            else:
                msg = 'Only csv file supported or unable to upload file'
                return render(request, 'home.html', {'msg':msg})
        return render(request, 'home.html', {'file_url': file_url,'msg':msg})
        
    elif request.method == 'GET':
        path = config('FOLDER_LOCATION')
        dir_list = os.listdir(path)
        return render(request, 'home.html',{'all_files':dir_list})
    else:
         return render(request, 'home.html')



def datatable(request, fileCSV):
    path = config('FOLDER_LOCATION')
    data = pd.read_csv(path+'/'+fileCSV, on_bad_lines='skip')#on_bad_lines='skip'    sep='delimiter'
    if(len(data) >= 100):
        first50 = data.head(20)
        last50 = data.tail(20)
        data1_html = first50.to_html()
        data2_html = last50.to_html()
        context = {'loaded_data1': data1_html,'loaded_data2':data2_html,'graph':fileCSV}#'graph':graph_dis
        return render(request, 'files.html', context)
    else:
        data1_html = data.to_html()
        context = {'loaded_data': data1_html,'graph':fileCSV}
        return render(request, 'files.html', context)

def displayGraph(request):
    path = config('FOLDER_LOCATION')
    dir_list = os.listdir(path)
    req_dataset = ['Latest Covid-19 India Status.csv', 'WHO-COVID-19-global-data.csv', 'covid_19_data.csv']
    data=''
    lat_one = "my_folder/images/lat_one.png"
    lat_two = "my_folder/images/lat_two.png"
    lat_three = "my_folder/images/lat_three.png"
    lat_four = "my_folder/images/lat_four.png"
    ld=''
    who_img = "my_folder/images/who_img.png"
    temp = "my_folder/images/temp.png"

    for i in dir_list:
        if(i not in req_dataset):
            data = pd.read_csv(path+'/'+i,on_bad_lines='skip')#on_bad_lines='skip'    sep='delimiter'
            ax = data.plot(color='#5D3FD3', figsize=(15,6))
            picture = ax.figure.savefig(temp)
        elif(i in req_dataset and i == 'Latest Covid-19 India Status.csv'):
            data = pd.read_csv(path+'/'+i, on_bad_lines='skip')#on_bad_lines='skip'    sep='delimiter'
            col = list(data.columns)
            if(col == ['State/UTs','Total Cases','Active','Discharged','Deaths','Active Ratio','Discharge Ratio','Death Ratio','Population']):
                state_population = data.sort_values(by=['Population'], ascending=False)
                plt.figure(figsize=(12,10))
                population_cnt = sns.barplot(x=state_population['Population'],y=state_population['State/UTs'], palette="Blues_d")
                dig = plt.title('State/UTs vs. Population')
                dig.figure.savefig(lat_one)

                # Sorting the list in descending order of total number of cases -
                total_cases_desc = data.sort_values(by=['Total Cases'], ascending=False)
                plt.figure(figsize=(12,8))
                max_cnt = sns.barplot(x=total_cases_desc['Total Cases'],y= total_cases_desc['State/UTs'], palette="Blues_d")
                dig2 = plt.title('Number of Total Cases (Statewise)')
                dig2.figure.savefig(lat_two)

                plt.figure(figsize=(12,8))
                max_cnt = sns.barplot(x=total_cases_desc['Total Cases'].head(15), y=total_cases_desc['State/UTs'].head(15), color='yellow')
                active_cases_cnt = sns.barplot(x=total_cases_desc['Active'].head(15), y=total_cases_desc['State/UTs'].head(15), color='red')
                dig3 = plt.title('Number of Active Cases (Statewise)')
                dig3.figure.savefig(lat_three)
                fig = px.pie(values=data['Deaths'], names=data['State/UTs'], title='Percent of deaths (Statewise)', width=1000, height=800, color_discrete_sequence=px.colors.sequential.Agsunset)
                fig.write_image(lat_four)
            else:
                pass
        elif(i in req_dataset and i == 'covid_19_data.csv'):
            data = pd.read_csv(path+'/'+i, on_bad_lines='skip')#on_bad_lines='skip'    sep='delimiter'
            col = list(data.columns)
            if(col ==['SNo', 'ObservationDate', 'Province/State', 'Country/Region', 'Last Update', 'Confirmed', 'Deaths', 'Recovered']):
                ld = lock(i)
            else:
                pass
        elif(i in req_dataset and i == 'WHO-COVID-19-global-data.csv'):  
            data = pd.read_csv(path+'/'+i, parse_dates= ['Date_reported'],on_bad_lines='skip')#on_bad_lines='skip'    sep='delimiter'
            data = pd.DataFrame(data[data['Country']=='India'][['Date_reported','New_cases']])
            data.set_index('Date_reported',inplace=True)
            ax = data.plot(color='#5D3FD3', figsize=(15,6))
            picture = ax.figure.savefig(who_img)
    return render(request,'charts.html',{'lat_one':lat_one,'lat_two':lat_two,'lat_three':lat_three,'lat_four':lat_four,'ld':ld,'who_img':who_img,'temp':temp})

    

def selectFile(request, f):
    global selected
    selected = f
    print('---------------------------')
    print(selected)
    print('---------------------------')
    return render(request,'home.html',{'selectedFileName':selected})

def models(request, temp):
    result=[]
    pic1=''
    pic2=''
    mae=''
    mse=''
    if request.method == 'GET' and selected:
        #temp = request.GET.get('fav_language')
        if temp == 'LSTM':
            result = lstmForecasting(selected)
            pic1 = result[0]
            pic2 = result[1]
            mae = result[2]
            mse = result[3]
            rmse = result[4]
            # rmse_vals.append(rmse)
            # mse_vals.append(mse)
            # mae_vals.append(mae)
            return render(request,'home.html',{'img1':pic1,'img2':pic2,'mae':mae,'mse':mse,'rmse':rmse})
        elif(temp == 'prophet'):
            result = prophetForecasting(selected)
            pic1 = result[0]
            pic2 = result[1]
            mae = result[2]
            mse = result[3]
            rmse = result[4]
            # rmse_vals.append(rmse)
            # mse_vals.append(mse)
            # mae_vals.append(mae)
            return render(request,'home.html',{'img1':pic1,'img2':pic2,'mae':mae,'mse':mse,'rmse':rmse})
        elif(temp == 'arima'):
            result = arimaForecasting(selected)
            pic1 = result[0]
            rmse = result[1]
            mse = result[2]
            mae = result[3]
            # rmse_vals.append(rmse)
            # mse_vals.append(mse)
            # mae_vals.append(mae)
            return render(request,'home.html',{'img1':pic1,'mae':mae,'mse':mse,'rmse':rmse})
        print(selected, temp)
        return render(request,'home.html')
    else:
        return render(request,'home.html',{'fileError':'File not selected'})

def metrics(request):
    #selected_models = request.GET.getlist('select_mod')
    met = "my_folder/images/metrics.png"
    models = ['LSTM Model','Prophet Model','ARIMA Model']
    #if(selected_models == models and selected):
    if(request.method == 'GET' and selected):
            #if('LSTM Model' in selected_models):
        lstm_result = lstmForecasting(selected)
        l_mae = lstm_result[2]
        l_mse = lstm_result[3]
        l_rmse = lstm_result[4]
        # rmse_vals.append(rmse)
        # mse_vals.append(mse)
        # mae_vals.append(mae)
            #if('Prophet Model' in selected_models):
        prophet_result = prophetForecasting(selected)
        p_mae = prophet_result[2]
        p_mse = prophet_result[3]
        p_rmse = prophet_result[4]
        #rmse_vals.append(rmse)
        #mse_vals.append(mse)
        #mae_vals.append(mae)
            #if('ARIMA Model' in selected_models):
        arima_result = arimaForecasting(selected)
        a_rmse = arima_result[1]
        a_mse = arima_result[2]
        a_mae = arima_result[3]
        # rmse_vals.append(rmse)
        # mse_vals.append(mse)
        # mae_vals.append(mae)
    elif(selected == ''):
        return render(request,'home.html',{'fileError':'File not selected'})
    global rmse_vals 
    rmse_vals = [l_rmse,p_rmse,a_rmse]
    global mae_vals
    mae_vals = [l_mae,p_mae,a_mae]
    global mse_vals
    mse_vals = [l_mse,p_mse,a_mse]
    model_evaluation=pd.DataFrame(list(zip(models,mse_vals,mae_vals,rmse_vals)),columns=["Model Name","Mean Squared Error","Mean Absolute Error","Root Mean Squared Error"])
    model_evaluation=model_evaluation.sort_values(["Root Mean Squared Error"])
    df_styled=model_evaluation.style.background_gradient(cmap='Reds')
    dfi.export(df_styled,met)
    return render(request,'home.html',{'met':met})

def deleteFile(request, fileName):
    path = config('FOLDER_LOCATION')
    imgpath = config('IMG_LOCATION')
    dir_list = os.listdir(path)
    img_list = os.listdir(imgpath)
    os.remove(path+'/'+fileName)
    global selected
    global rmse_vals
    global mse_vals
    global mae_vals
    if(selected == fileName):
        selected=''
        rmse_vals = []
        mse_vals = []
        mae_vals = []
    
    if(fileName == 'WHO-COVID-19-global-data.csv'):
        if('who_img.png' in img_list):
            os.remove(imgpath+'/'+'who_img.png')
    if(fileName == 'covid_19_data.csv'):
        if('lock.png' in img_list):
            os.remove(imgpath+'/'+'lock.png')
    if(fileName == 'Latest Covid-19 India Status.csv'):
        if('lat_one.png' in img_list):
            os.remove(imgpath+'/'+'lat_one.png')
        if('lat_two.png' in img_list):
            os.remove(imgpath+'/'+'lat_two.png')
        if('lat_three.png' in img_list):
            os.remove(imgpath+'/'+'lat_three.png')
        if('lat_four.png' in img_list):
            os.remove(imgpath+'/'+'lat_four.png')
    if(dir_list == [] and img_list != []):
        for i in img_list:
            print(i)
            os.remove(imgpath+'/'+i)
    return render(request,'home.html')

