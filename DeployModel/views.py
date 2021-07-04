from django.http import HttpResponse
from django.shortcuts import render
import pandas as pd
import joblib


# Create your views here.
def home(request):
    return render(request, "home.html")

def result(request):
    LA = joblib.load('finalized_model.sav')

    s1 = pd.Series([(request.GET['Tenure']),(request.GET['Dependents']),(request.GET['MultipleLines']),(request.GET['InternetService']),(request.GET['PhoneService']),(request.GET['PaymentMethod']),(request.GET['TotalCharges']),(request.GET['Contract']),(request.GET['StreamingTV']),(request.GET['OnlineBackup'])])
    cols =  ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J"]

    df = pd.DataFrame([list(s1)],  columns =  cols)

    # lis = []

    # lis.append(request.GET['Tenure'])
    # lis.append(request.GET['Dependents'])
    # lis.append(request.GET['MultipleLines'])
    # lis.append(request.GET['InternetService'])
    # lis.append(request.GET['PhoneService'])
    # lis.append(request.GET['PaymentMethod'])
    # lis.append(request.GET['TotalCharges'])
    # lis.append(request.GET['Contract'])
    # lis.append(request.GET['StreamingTV'])
    # lis.append(request.GET['OnlineBackup'])

    ans = LA.predict(df)

    return render(request,"result.html",{'ans':ans})