Urls.py
from django.contrib import admin
from django.urls import path
from users import views as users
from admins import views as admins
from django.conf import settings
from django.conf.urls.static import static

urlpatterns = [
    path('admin/', admin.site.urls),
    path('index/', users.index, name='index'),
    path('UserLogin/', users.UserLogin, name='UserLogin'),
    path('UserRegister/', users.UserRegister, name='UserRegister'),
    path('UserRegisterAction/', users.UserRegisterAction, name='UserRegisterAction'),
    path('UserLoginCheck/', users.UserLoginCheck, name='UserLoginCheck'),
    path('UserSendCrop/', users.UserSendCrop, name='UserSendCrop'),
    path('UserSendCropanalysis/', users.UserSendCropanalysis, name='UserSendCropanalysis'),
    path('yeilddetails/', users.yeilddetails, name='yeilddetails'),
    path('ML/', users.ML, name='ML'),


    path('AdminLogin/', admins.AdminLogin, name='AdminLogin'),
    path('AdminLoginCheck/', admins.AdminLoginCheck, name='AdminLoginCheck'),
    path('AdminViewUsers/', admins.AdminViewUsers, name='AdminViewUsers'),
    path('AdminActivaUsers/', admins.AdminActivaUsers, name='AdminActivaUsers'),

    path('Sendcropdata/', admins.Sendcropdata, name='Sendcropdata'),
    path('sendcrop/', admins.sendcrop, name='sendcrop'),
    path('storecsvdata/', admins.storecsvdata, name='storecsvdata'),
    path('MLprocess/', admins.MLprocess, name='MLprocess'),
]
if settings.DEBUG:
    urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)
Users views.py
from django.contrib import messages
from django.shortcuts import render

# Create your views here.
from django_pandas.io import read_frame
from nltk.corpus import wordnet

from admins.models import storedatamodel
from users.forms import UserRegistrationForm
from users.models import cropyieldUserRegistrationModel, cropyieldanalysismodel
import datetime
import time
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix
from matplotlib.colors import ListedColormap
import numpy as nm
import pandas as pd
import matplotlib.pyplot as mtp


def index(request):
    return render(request,'index.html')


def UserLogin(request):
    return render(request,'users/UserLogin.html',{})


def UserRegister(request):
    form = UserRegistrationForm()
    return render(request,'users/UserRegisterForm.html',{'form':form})


def UserRegisterAction(request):
    if request.method == 'POST':
        form = UserRegistrationForm(request.POST)
        if form.is_valid():

            print('Data is Valid')
            form.save()
            messages.success(request, 'You have been successfully registered')
            # return HttpResponseRedirect('./CustLogin')
            form = UserRegistrationForm()
            return render(request, 'users/UserRegisterForm.html', {'form': form})
        else:
            print("Invalid form")
    else:
        form = UserRegistrationForm()
    return render(request, 'users/UserRegisterForm.html', {'form': form})


def UserLoginCheck(request):
    if request.method == "POST":
        loginid = request.POST.get('loginid')
        pswd = request.POST.get('pswd')
        print("Login ID = ", loginid, ' Password = ', pswd)
        try:
            check = cropyieldUserRegistrationModel.objects.get(loginid=loginid, password=pswd)
            status = check.status
            print('Status is = ', status)
            if status == "activated":
                request.session['id'] = check.id
                request.session['loggeduser'] = check.name
                request.session['loginid'] = loginid
                request.session['email'] = check.email
                print("User id At", check.id, status)
                return render(request, 'users/UserHomePage.html', {})
            else:
                messages.success(request, 'Your Account Not at activated')
                return render(request, 'users/UserLogin.html')
            # return render(request, 'user/userpage.html',{})
        except Exception as e:
            print('Exception is ', str(e))
            pass
        messages.success(request, 'Invalid Login id and password')
    return render(request, 'users/UserLogin.html', {})


def UserSendCrop(request):
    return render(request,'users/UserSendCrop.html')


class NaiveBayes:
    def __init__(self, name, crops):
        self.name = name

        self.crops = crops

known_yeilds = [
    NaiveBayes('116.58', set("Rice|andhrapradesh|Kharif".split("|"))),
    NaiveBayes('131.68', set("Rice|westbengal|Kharif".split("|"))),
    NaiveBayes('116.20', set("Rice|uttarpradesh|Kharif".split("|"))),
    NaiveBayes('86.03', set("Rice|punjab|Kharif".split("|"))),
    NaiveBayes('55.97', set("Rice|bihar|Kharif".split("|"))),
    NaiveBayes('50.52', set("Rice|orissa|Kharif".split("|"))),
    NaiveBayes('41.44', set("Rice|chhattisgarh|Kharif".split("|"))),
    NaiveBayes('17.92', set("Rice|assam|Kharif".split("|"))),
    NaiveBayes('76.31', set("Rice|tamilnadu|Kharif".split("|"))),
    NaiveBayes('25.68', set("Rice|haryana|Kharif".split("|"))),
    NaiveBayes('3,322', set("Rice|westgodawari|Kharif".split("|"))),
    NaiveBayes('3,239', set("Rice|guntur|Kharif".split("|"))),
    NaiveBayes('3,142', set("Rice|krishna|Kharif".split("|"))),
    NaiveBayes('2,985', set("Rice|prakasham|Kharif".split("|"))),
    NaiveBayes('2,978', set("Rice|eastgodavari|Kharif".split("|"))),
    NaiveBayes('2,942', set("Rice|kurnool|Kharif".split("|"))),
    NaiveBayes('2,864', set("Rice|nellore|Kharif".split("|"))),
    NaiveBayes('2,630', set("Rice|anantpur|Kharif".split("|"))),
    NaiveBayes('2,610', set("Rice|cuddapah|Kharif".split("|"))),
    NaiveBayes('2,373', set("Rice|chittor|Kharif".split("|"))),
    NaiveBayes('1,957', set("Rice|vizianagaram|Kharif".split("|"))),
    NaiveBayes('1,864', set("Rice|srikakulam|Kharif".split("|"))),
    NaiveBayes('1,430', set("Rice|vishakhapatnam|Kharif".split("|"))),
    NaiveBayes('2,803', set("Rice|karimnagar|Kharif".split("|"))),
    NaiveBayes('2,678', set("Rice|nizamabad|Kharif".split("|"))),
    NaiveBayes('2,578', set("Rice|khammam|Kharif".split("|"))),
    NaiveBayes('3,206', set("Rice|nalgonda|Kharif".split("|"))),
    NaiveBayes('2,398', set("Rice|medak|Kharif".split("|"))),
    NaiveBayes('2,321', set("Rice|rangareddy|Kharif".split("|"))),
    NaiveBayes('2,320', set("Rice|adilabad|Kharif".split("|"))),
    NaiveBayes('3,462', set("Rice|koppal|Kharif".split("|"))),
    NaiveBayes('3,379', set("Rice|davangere|Kharif".split("|"))),
    NaiveBayes('3,247', set("Rice|bellary|Kharif".split("|"))),
    NaiveBayes('2,993', set("Rice|mysore|Kharif".split("|"))),
    NaiveBayes('2,851', set("Rice|raichur|Kharif".split("|"))),
    NaiveBayes('2,749', set("Rice|bangalore|Kharif".split("|"))),
    NaiveBayes('637', set("Rice|bidar|Kharif".split("|"))),
    NaiveBayes('4,574', set("Rice|madurai|Kharif".split("|"))),
    NaiveBayes('4,434', set("Rice|thirunelveli|Kharif".split("|"))),
    NaiveBayes('3,769', set("Rice|vellore|Kharif".split("|"))),



]



def UserSendCropanalysis(request):
    if request.method == "POST":
        crops = request.POST.get('crop')
        print(crops)
        loginid = request.POST.get('loginid')
        print(loginid)
        try:

            check = cropyieldUserRegistrationModel.objects.get(loginid=loginid)
            loginid = check.loginid
            print("name", loginid)
            email = check.email
            storcrops = crops
            print(check.email, storcrops)
            crops = crops.lower()
            #print("crops:",crops)
            crops = crops.split(",")
            possible = []
            for crop in crops:
                #print("crop",crop)
                for yeilds in known_yeilds:
                    #print("yeilds",yeilds)
                    if crop in yeilds.crops:
                        possible.append(yeilds.name)
            if possible:
                print("possible",possible)
                for x in possible:
                    print('yeild is = ', x)
                    #recDescription = recDesc[x]

                    ing = wordnet.synsets(x)
                    description = ''
                    if len(ing) != 0:
                        description = ing[0].definition()
                        print(description)
                    else:
                        description = 'No Data found'
                        cropyieldanalysismodel.objects.create(loginid=loginid, email=email, cropdetails=storcrops,
                                                              yields=x)
                messages.success(request, 'Your Request Sent to admin')
            else:
                messages.success(request, "Sorry,Based on details we can't provide proper deatil")
            return render(request, 'users/UserSendCrop.html')
        except Exception  as e:

            print(str(e))

        messages.success(request, 'There is a problam in your details')
        return render(request, 'users/UserSendCrop.html')


def yeilddetails(request):
    email = request.session['email']
    sts = 'sent'
    dict = cropyieldanalysismodel.objects.filter(email=email,status=sts).order_by('-id')
    return render(request,'users/yeildsdetails.html',{'data':dict})


def ML(request):
    qs = storedatamodel.objects.all()
    data = read_frame(qs)
    data = data.fillna(data.mean())
    # data[0:label]
    data.info()
    print(data.head())
    print(data.describe())
    #print(data.shape)
    # print("data-label:",data.label)
    dataset = data.iloc[:,[3,4]].values
    print("x", dataset)
    dataset1 = data.iloc[:,-1].values
    print("y", dataset1)
    print("shape", dataset.shape)
    X = dataset
    y = dataset1
    print(dataset.shape)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=1 / 3, random_state=0)
    st_X = StandardScaler()
    X_train = st_X.fit_transform(X_train)
    X_test = st_X.transform(X_test)
    classifier = KNeighborsClassifier()
    classifier.fit(X_train, y_train)

    y_pred = classifier.predict(X_test)
    print("y_pred", y_pred)
    cm = confusion_matrix(y_test, y_pred)
    print("cm", cm)
    X_set, y_set = X_train, y_train
    X1, X2 = nm.meshgrid(nm.arange(start=X_set[:, 0].min() - 1, stop=X_set[:, 0].max() + 1, step=0.01),
                        


 nm.arange(start=X_set[:, 1].min() - 1, stop=X_set[:, 1].max() + 1, step=0.01))
    mtp.contourf(X1, X2, classifier.predict(nm.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
                 alpha=0.75, cmap=ListedColormap(('red', 'green')))
    mtp.xlim(X1.min(), X1.max())
    mtp.ylim(X2.min(), X2.max())
    for i, j in enumerate(nm.unique(y_set)):
        mtp.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
                    c=ListedColormap(('red', 'green'))(i), label=j)
    mtp.title('K-NN Algorithm (Training set)')
    mtp.xlabel('Year')
    mtp.ylabel('Estimated yeild')
    mtp.legend()
    mtp.show()
    return render(request, 'users/UserHomePage.html')
Admin views.py
from django.contrib import messages
from django.shortcuts import render
from io import TextIOWrapper
import csv
from collections import defaultdict
from django.shortcuts import render, HttpResponse
# Create your views here.
from django_pandas.io import read_frame

from admins.models import  storedatamodel
from users.models import cropyieldUserRegistrationModel, cropyieldanalysismodel
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix
from matplotlib.colors import ListedColormap
import numpy as nm
import pandas as pd
import matplotlib.pyplot as mtp
from sklearn.metrics import classification_report


def AdminLogin(request):
    return render(request,'admins/AdminLogin.html',{})

def AdminLoginCheck(request):
    if request.method == 'POST':
        usrid = request.POST.get('loginname')
        pswd = request.POST.get('pswd')
        print("User ID is = ", usrid)
        if usrid == 'admin' and pswd == 'admin@2020':
            return render(request, 'admins/AdminHome.html')

        else:
            messages.success(request, 'Please Check Your Login Details')
    return render(request, 'admins/AdminLogin.html', {})


def AdminViewUsers(request):
    data = cropyieldUserRegistrationModel.objects.all()
    return render(request,'admins/AdminViewUsers.html',{'data':data})

def AdminActivaUsers(request):
    if request.method == 'GET':
        id = request.GET.get('uid')
        status = 'activated'
        print("PID = ", id, status)
        cropyieldUserRegistrationModel.objects.filter(id=id).update(status=status)
        data = cropyieldUserRegistrationModel.objects.all()
        return render(request,'admins/AdminViewUsers.html',{'data':data})


def Sendcropdata(request):
    data = cropyieldanalysismodel.objects.all()
    return render(request,'admins/AdminViewcropdetails.html',{'data':data})

def sendcrop(request):
    if request.method == 'GET':
        id = request.GET.get('id')
        print(' ID = ', id)
        loginid = request.session['loginid']
        cropyieldanalysismodel.objects.filter(id=id).update(status='sent')
        data = cropyieldanalysismodel.objects.filter(loginid=loginid)
        return render(request, 'admins/AdminViewcropdetails.html', {'data': data})




def storecsvdata(request):
    if request.method == 'POST':
        name = request.POST.get('name')
        csvfile =TextIOWrapper( request.FILES['file'])
        columns = defaultdict(list)

        storecsvdata = csv.DictReader(csvfile)

        for row1 in storecsvdata:
                state = row1["state"]
                dist = row1["dist"]
                yeild = row1["yeild"]

                year = row1["year"]
                label = row1["label"]

                storedatamodel.objects.create(state=state, dist=dist, yeild=yeild,
                                                year=year, label=label)

        print("Name is ",csvfile)
        return HttpResponse('CSV file successful uploaded')
    else:

        return render(request, 'admins/storecropdata.html', {})


def MLprocess(request):
    qs = storedatamodel.objects.all()
    data = read_frame(qs)
    data = data.fillna(data.mean())
    # data[0:label]
    data.info()
    print(data.head())
    print(data.describe())
    #print(data.shape)
    # print("data-label:",data.label)
    dataset = data.iloc[:,[3,4]].values
    print("x", dataset)
    dataset1 = data.iloc[:,-1].values
    print("y", dataset1)
    print("shape", dataset.shape)
    X = dataset
    y = dataset1
    print(dataset.shape)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=1 / 3, random_state=0)
    st_X = StandardScaler()
    X_train = st_X.fit_transform(X_train)
    X_test = st_X.transform(X_test)
    classifier = KNeighborsClassifier()
    classifier.fit(X_train, y_train)

    y_pred = classifier.predict(X_test)
    print("y_pred", y_pred)
    cm = confusion_matrix(y_test, y_pred)
    print("cm", cm)
    accurancy = classifier.score(X_train, y_train)
    print("accurancy", accurancy)
    predicition =classification_report(y_test, y_pred)
    print("predicition",predicition)
    x = predicition.split()

    print("Toctal splits ", len(x))
    X_set, y_set = X_train, y_train
    X1, X2 = nm.meshgrid(nm.arange(start=X_set[:, 0].min() - 1, stop=X_set[:, 0].max() + 1, step=0.01),
                         nm.arange(start=X_set[:, 1].min() - 1, stop=X_set[:, 1].max() + 1, step=0.01))
    mtp.contourf(X1, X2, classifier.predict(nm.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
                 alpha=0.75, cmap=ListedColormap(('red', 'green')))
    mtp.xlim(X1.min(), X1.max())
    mtp.ylim(X2.min(), X2.max())
    for i, j in enumerate(nm.unique(y_set)):
        mtp.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
                    c=ListedColormap(('red', 'green'))(i), label=j)
    mtp.title('K-NN Algorithm (Training set)')
    mtp.xlabel('Year')
    mtp.ylabel('Estimated yeild')
    mtp.legend()
    mtp.show()

    dict = {
        "accurancy": accurancy,
        #"predicition":predicition,
        'len0': x[0],
        'len1': x[1],
        'len2': x[2],
        'len3': x[3],
        'len4': x[4],
        'len5': x[5],
        'len6': x[6],
        'len7': x[7],
        'len8': x[8],
        'len9': x[9],
        'len10': x[10],
        'len11': x[11],
        'len12': x[12],
        'len13': x[13],
        'len14': x[14],
        'len15': x[15],
        'len16': x[16],
        'len17': x[17],
        'len18': x[18],
        'len19': x[19],
        'len20': x[20],
        'len21': x[21],
        'len22': x[22],
        'len23': x[23],
        'len24': x[24],
        'len25': x[25],
        'len26': x[26],
        'len27': x[27],
        'len28': x[28],

    }

    return render(request, 'admins/mlaccurancy.html',dict)
Model.py
from django.db import models

# Create your models here.

class cropyieldUserRegistrationModel(models.Model):
    name = models.CharField(max_length=100)
    loginid = models.CharField(unique=True,max_length=100)
    password = models.CharField(max_length=100)
    mobile = models.CharField(max_length=100)
    email = models.CharField(max_length=100)
    locality = models.CharField(max_length=100)
    address = models.CharField(max_length=1000)
    city = models.CharField(max_length=100)
    state = models.CharField(max_length=100)
    status  = models.CharField(max_length=100)

    def __str__(self):
        return self.loginid
    class Meta:
        db_table='cropyieldUsers'


class cropyieldanalysismodel(models.Model):
    #id = models.AutoField(primary_key=True)
    loginid = models.CharField(max_length=100)
    email = models.EmailField()
    cropdetails = models.CharField(max_length=600)
    yields = models.CharField(max_length=250)
    #descriptions = models.CharField(max_length=600)
    status = models.CharField(max_length=600, default='waiting')
    name = models.CharField(max_length=600, default='notassigned')

    def __str__(self):
        return self.email
    class Meta:
        db_table='cropyielddata'

Forms.py
from django import forms


from users.models import cropyieldUserRegistrationModel,  cropyieldanalysismodel


class UserRegistrationForm(forms.ModelForm):
    name = forms.CharField(widget=forms.TextInput(attrs={'pattern':'[a-zA-Z]+'}), required=True,max_length=100)
    loginid = forms.CharField(widget=forms.TextInput(attrs={'pattern':'[a-zA-Z]+'}), required=True,max_length=100)


    password = forms.CharField(widget=forms.PasswordInput(attrs={'pattern':'(?=.*\d)(?=.*[a-z])(?=.*[A-Z]).{8,}','title':'Must contain at least one number and one uppercase and lowercase letter, and at least 8 or more characters'}), required=True,max_length=100)
    mobile = forms.CharField(widget=forms.TextInput(attrs={'pattern':'[56789][0-9]{9}'}), required=True,max_length=100)
    email = forms.CharField(widget=forms.TextInput(attrs={'pattern':'[a-z0-9._%+-]+@[a-z0-9.-]+\.[a-z]{2,}$'}), required=True,max_length=100)
    locality = forms.CharField(widget=forms.TextInput(), required=True,max_length=100)
    address = forms.CharField(widget=forms.Textarea(attrs={'rows':4, 'cols': 22}), required=True,max_length=250)
    city = forms.CharField(widget=forms.TextInput(attrs={'class':'form-control' , 'autocomplete': 'off','pattern':'[A-Za-z ]+', 'title':'Enter Characters Only '}), required=True,max_length=100)
    state = forms.CharField(widget=forms.TextInput(attrs={'class':'form-control' , 'autocomplete': 'off','pattern':'[A-Za-z ]+', 'title':'Enter Characters Only '}), required=True,max_length=100)
    status = forms.CharField(widget=forms.HiddenInput(), initial='waiting' ,max_length=100)


    class Meta():
        model = cropyieldUserRegistrationModel
        fields='__all__'



class cropyieldanalysisform(forms.ModelForm):
    class Meta:
        model = cropyieldanalysismodel
        fields = ['loginid','email','cropdetails','yields','status','name']
