from django.shortcuts import render, redirect
from django.contrib.auth.models import User 
# Create your views here.
from django.contrib import messages
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score,classification_report,confusion_matrix
from sklearn.model_selection import train_test_split
from collections.abc import Iterable
from . models import *


def index(request):

    return render(request,'index.html')

def about(request):
    
    return render(request,'about.html')


def login(request):
    if request.method=='POST':
        lemail=request.POST['email']
        lpassword=request.POST['password']

        d=Register.objects.filter(email=lemail,password=lpassword).exists()
        print(d)
        return redirect('userhome')
    else:
        return render(request,'login.html')

def registration(request):
    if request.method=='POST':
        Name = request.POST['Name']
        email=request.POST['email']
        password=request.POST['password']
        conpassword=request.POST['conpassword']
        

        print(Name,email,password,conpassword)
        if password==conpassword:
            rdata=Register(name=Name,email=email,password=password)
            rdata.save()
            return render(request,'login.html')
        else:
            msg='Register failed!!'
            return render(request,'registration.html')

    return render(request,'registration.html')
    # return render(request,'registration.html')


def userhome(request):
    
    return render(request,'userhome.html')

def load(request):
   if request.method=="POST":
        file=request.FILES['file']
        global df
        df=pd.read_csv(file)
        messages.info(request,"Data Uploaded Successfully")
    
   return render(request,'load.html')

def view(request):
    col=df.to_html
    dummy=df.head(100)
   
    col=dummy.columns
    rows=dummy.values.tolist()
    return render(request, 'view.html',{'col':col,'rows':rows})

    # return render(request,'viewdata.html', {'columns':df.columns.values, 'rows':df.values.tolist()})
    
    
def preprocessing(request):

    global X_train,X_test,y_train,y_test,X,y
    if request.method == "POST":
        # size = request.POST['split']
        size = int(request.POST['split'])
        size = size / 100
        df.drop('date',axis=1,inplace=True)
        from sklearn.preprocessing import LabelEncoder
        le = LabelEncoder()
        df['weather'] = le.fit_transform(df['weather'])

        #Preprocess Data for Machine Learning Development
        X = df.drop(['weather'], axis = 1)
        y = df['weather']


        over_strategy = {0 : 1000, 1 : 1000, 2 : 1000, 3 : 1000, 4 : 2000}
        under_strategy = {0 : 1000, 1 : 1000, 2 : 1000, 3 : 1000, 4 : 2000}

        oversample = SMOTE(sampling_strategy = over_strategy)
        undersample = RandomUnderSampler(sampling_strategy = under_strategy)

        X_final,y = oversample.fit_resample(X,y)
        X_final,y = undersample.fit_resample(X_final, y)


        X_train,X_test,y_train,y_test = train_test_split(X_final,y,random_state = 10, test_size = 0.2)

        messages.info(request,"Data Preprocessed and It Splits Succesfully")
   
    return render(request,'preprocessing.html')
 



def model(request):
    if request.method == "POST":

        model = request.POST['algo']

        if model == "0":
            from sklearn.ensemble import RandomForestClassifier
            rf = RandomForestClassifier(n_estimators=52)
            rf = rf.fit(X_train,y_train)
            y_pred = rf.predict(X_test)
            acc_rf=accuracy_score(y_test,y_pred)
            msg = 'Accuracy of RandomForestClassifier : ' + str(acc_rf)
            return render(request,'model.html',{'msg':msg})
        elif model == "1":
            from sklearn.tree import DecisionTreeClassifier 
            dt = DecisionTreeClassifier(criterion="entropy",max_depth=3,random_state=1245)
            dt = dt.fit(X_train,y_train)
            y_pred = dt.predict(X_test)
            acc_dt=accuracy_score(y_test,y_pred)
            msg = 'Accuracy of DecisionTreeClassifier :  ' + str(acc_dt)
            return render(request,'model.html',{'msg':msg})
        elif model == "2":
            from sklearn.linear_model import LogisticRegression
            lr = LogisticRegression(random_state=100)
            lr = lr.fit(X_train,y_train)
            y_pred = lr.predict(X_test)
            acc_lr=accuracy_score(y_test,y_pred)
            msg = 'Accuracy of LogisticRegression :  ' + str(acc_lr)
            return render(request,'model.html',{'msg':msg})     
    return render(request,'model.html')

def prediction(request):

    global X_train,X_test,y_train,y_test,X,y
    

    if request.method == 'POST':

        f1 = float(request.POST['precipitation'])
        f2 = float(request.POST['temp_max'])
        f3 = float(request.POST['temp_min'])
        f4 = float(request.POST['wind'])


        PRED = [[f1,f2,f3,f4]]
       
        from sklearn.tree import DecisionTreeClassifier 
        model = DecisionTreeClassifier()
        model.fit(X_train,y_train)
        xgp = np.array(model.predict(PRED))

        if xgp==0:
            msg = ' <span style = color:white;>The Weather is going to be : <span style = color:green;><b>Drizzle</b></span></span>'
        elif xgp==1:
            msg = ' <span style = color:white;>The Weather is going to be: <span style = color:red;><b>Fog</b></span></span>'
        elif xgp==2:
            msg = ' <span style = color:white;>The Weather is going to be: <span style = color:red;><b>Rain</b></span></span>'
        elif xgp==3:
            msg = ' <span style = color:white;>The Weather is going to be: <span style = color:red;><b>Snow</b></span></span>'
        else :
            msg = ' <span style = color:white;>The Weather is going to be: <span style = color:red;><b>Sun</b></span></span>'
        
        return render(request,'prediction.html',{'msg':msg})

    
    return render(request,'prediction.html')