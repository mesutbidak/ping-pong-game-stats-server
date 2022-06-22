#from crypt import methods
from distutils.log import info
import string
from xml.dom.minidom import TypeInfo
from flask import Flask,request,render_template,jsonify
import json
import numpy as np #yüklü olmayabilir kontrol et.
import pickle #buda yüklü olmayabilir
import pandas as pd
# from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from ab_test import *

tempDf3 = pd.DataFrame()

pd.set_option('display.max_columns', None)
pd.set_option('display.width', 500)

import os
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 500)
path = "txts"

files = [file for file in os.listdir(path) if not file.startswith('.')]



listem=[["Female","Female",67, 60,165, 160, 19,"right_hand", "left_hand",20, 22,1,2,"train"],
["Male","Female", 85,53, 190,160, 22, "right_hand","right_hand",21, 21,7,0,"train"],
["Female","Male", 62,69, 178,171, 20, "right_hand","left_hand", 23,19,1,3,"train"],
["Male","Male",83, 78, 185,181, 11, "right_hand","left_hand",21, 22,3,4,"train"],
["Male","Female", 80,59, 187,173, 21, "left_hand","right_hand",22, 23,4,1,"train"],
["Female","Male", 66, 78, 169,178, 9, "right_hand","left_hand",27, 24,5,2,"train"],
["Male","Male", 70,80, 170,180, 10, "left_hand","right_hand",20, 25,2,3,"train"],
["Male","Female",83,72, 176,179, 23, "right_hand","right_hand", 39,25,4,0,"train"],
["Male","Male", 80,83, 187,186, 21, "left_hand","left_hand", 23,22,4,2,"train"],
["Female","Male", 74,92, 172,179, 20, "right_hand","right_hand", 26,24,6,2,"train"]]

def feature2(dataframe,sex_h,sex_a,weight_h,weight_a,height_h,height_a,clock,left_hand_h,left_hand_a,age_h,age_a,year_played_h,year_played_a,data_type):
    dataframe["Sex_Home"] = sex_h
    dataframe["Sex_Away"] = sex_a
    dataframe["Weight_Home"] = weight_h
    dataframe["Weight_Away"] = weight_a
    dataframe["Height_Home"] = height_h
    dataframe["Height_Away"] = height_a
    dataframe["Clock"] = clock
    dataframe["Left_hand_Home"] = left_hand_h
    dataframe["Left_hand_Away"] = left_hand_a
    dataframe["Age_Home"] = age_h
    dataframe["Age_Away"] = age_a
    dataframe["Year_Played_Home"] = year_played_h
    dataframe["Year_Played_Away"] = year_played_a
    dataframe["BMI_Home"] = dataframe["Weight_Home"] / ((dataframe["Height_Home"]/100)**2)
    dataframe["BMI_Away"] = dataframe["Weight_Away"] / ((dataframe["Height_Away"]/100)**2)
    dataframe["Total_Piezzo"] = dataframe["piezzo1"] + dataframe["piezzo2"] + dataframe["piezzo3"] + dataframe["piezzo4"]
    dataframe["Data_Type"] = data_type
    # dataframe.loc[(dataframe["BMI_Home"] < 20), "BMI_CAT_Home"] = "Thin"
    # dataframe.loc[((dataframe["BMI_Home"] >= 20) & (dataframe["BMI_Home"] < 25)), 'BMI_CAT_Home'] = "Normal"
    # dataframe.loc[(dataframe["BMI_Home"] >= 25), "BMI_CAT_Home"] = "Fat"
    # dataframe.loc[(dataframe["BMI_Away"] < 20), "BMI_CAT_Away"] = "Thin"
    # dataframe.loc[((dataframe["BMI_Away"] >= 20) & (dataframe["BMI_Away"] < 25)), 'BMI_CAT_Away'] = "Normal"
    # dataframe.loc[(dataframe["BMI_Away"] >= 25), "BMI_CAT_Away"] = "Fat"
    # dataframe.loc[(dataframe["Age_Home"] < 18), "AGE_CAT_Home"] = "Child"
    # dataframe.loc[((dataframe["Age_Home"] >= 18) & (dataframe["Age_Home"] < 30)), 'AGE_CAT_Home'] = "Young"
    # dataframe.loc[(dataframe["Age_Home"] >= 30), "AGE_CAT_Home"] = "Adult"
    # dataframe.loc[(dataframe["Year_Played_Home"] < 3), "Level_Home"] = "Beginner"
    # dataframe.loc[((dataframe["Year_Played_Home"] >= 3) & (dataframe["Year_Played_Home"] < 5)), 'Level_Home'] = "Mid"
    # dataframe.loc[(dataframe["Year_Played_Home"] >= 5), "Level_Home"] = "Professional"
    # dataframe.loc[(dataframe["Year_Played_Away"] < 3), "Level_Away"] = "Beginner"
    # dataframe.loc[((dataframe["Year_Played_Away"] >= 3) & (dataframe["Year_Played_Away"] < 5)), 'Level_Away'] = "Mid"
    # dataframe.loc[(dataframe["Year_Played_Away"] >= 5), "Level_Away"] = "Professional"
    return dataframe
#sex_h,sex_a,weight_h,weight_a,height_h,height_a,clock,left_hand_h,left_hand_a,age_h,age_a,year_played_h,year_played_a

# listem=[["Female","Female",67, 60,165, 160, 19,"right_hand", "left_hand",20, 22,1,2],
# ["Male","Female", 85,53, 190,160, 22, "right_hand","right_hand",21, 21,7,0],
# ["Female","Male", 62,69, 178,171, 20, "right_hand","left_hand", 23,19,1,3],
# ["Male","Male",83, 78, 185,181, 11, "right_hand","left_hand",21, 22,3,4],
# ["Male","Female", 80,59, 187,173, 21, "left_hand","right_hand",22, 23,4,1],
# ["Female","Male", 66, 78, 169,178, 9, "right_hand","left_hand",27, 24,5,2],
# ["Male","Male", 70,80, 170,180, 10, "left_hand","right_hand",20, 25,2,3],
# ["Male","Female",83,72, 176,179, 23, "right_hand","right_hand", 39,25,4,0],
# ["Male","Male", 80,83, 187,186, 21, "left_hand","left_hand", 23,22,4,2],
# ["Female","Male", 74,92, 172,179, 20, "right_hand","right_hand", 26,24,6,2]]
df2=pd.DataFrame()
for iter,file in enumerate(files):
    current_data = pd.read_csv(path+"/"+file , header=None,sep=" ")
    headers = ["piezzo1", "piezzo2", "piezzo3", "piezzo4"]
    current_data.columns = headers
    current_data = feature2(current_data, *listem[iter])
    df2 = pd.concat([df2,current_data])
#######################################################################################

def feature99(dataframe,sex_h,sex_a,weight_h,weight_a,height_h,height_a,clock,left_hand_h,left_hand_a,age_h,age_a,year_played_h,year_played_a):
    dataframe["Weight_Home"] = weight_h
    dataframe["Weight_Away"] = weight_a
    dataframe["Height_Home"] = height_h
    dataframe["Height_Away"] = height_a
    dataframe["Age_Home"] = age_h
    dataframe["Age_Away"] = age_a
    dataframe["Year_Played_Home"] = year_played_h
    dataframe["Year_Played_Away"] = year_played_a
    dataframe["BMI_Home"] = dataframe["Weight_Home"] / ((dataframe["Height_Home"]/100)**2)
    dataframe["BMI_Away"] = dataframe["Weight_Away"] / ((dataframe["Height_Away"]/100)**2)
    dataframe["Total_Piezzo"] = dataframe["piezzo1"] + dataframe["piezzo2"] + dataframe["piezzo3"] + dataframe["piezzo4"]
    if sex_h == "Male":
        dataframe["Sex_Home_Male"] = 1
    if sex_h == "Female":
        dataframe["Sex_Home_Male"] = 0
    if sex_a == "Male":
        dataframe["Sex_Away_Male"] = 1
    if sex_a == "Female":
        dataframe["Sex_Away_Male"] = 0
    if clock <=12:
        dataframe["Clock_morning"] = 1
    if clock > 12:
        dataframe["Clock_morning"] = 0
    if left_hand_h == "Right Hand":
        dataframe["Left_hand_Home_right_hand"] = 1
    if left_hand_h == "Left Hand":
        dataframe["Left_hand_Home_right_hand"] = 0
    if left_hand_a == "Right Hand":
        dataframe["Left_hand_Away_right_hand"] = 1
    if left_hand_a == "Left Hand":
        dataframe["Left_hand_Away_right_hand"] = 0
    dataframe.loc[((dataframe["BMI_Home"] >= 20) & (dataframe["BMI_Home"] < 25)), 'BMI_CAT_Home_Normal'] = 1
    dataframe.loc[(dataframe["BMI_Home"] < 20), "BMI_CAT_Home_Thin"] = 1
    dataframe.loc[((dataframe["BMI_Away"] >= 20) & (dataframe["BMI_Away"] < 25)), 'BMI_CAT_Away_Normal'] = 1
    dataframe.loc[(dataframe["BMI_Away"] < 20), "BMI_CAT_Away_Thin"] = 1
    dataframe.loc[((dataframe["Age_Home"] >= 18) & (dataframe["Age_Home"] < 30)), 'AGE_CAT_Home_Young'] = 1
    dataframe.loc[((dataframe["Age_Away"] >= 18) & (dataframe["Age_Away"] < 30)), 'AGE_CAT_Away_Young'] = 1
    dataframe.loc[((dataframe["Year_Played_Home"] >= 3) & (dataframe["Year_Played_Home"] < 5)), 'Level_Home_Mid'] = 1
    dataframe.loc[(dataframe["Year_Played_Home"] >= 5), "Level_Home_Professional"] = 1
    dataframe.loc[((dataframe["Year_Played_Away"] >= 3) & (dataframe["Year_Played_Away"] < 5)), 'Level_Away_Mid'] = 1
    dataframe.loc[(dataframe["Year_Played_Away"] >= 5), "Level_Away_Professional"] = 1
    return dataframe

def feature(dataframe,sex_h,sex_a,weight_h,weight_a,height_h,height_a,clock,left_hand_h,left_hand_a,age_h,age_a,year_played_h,year_played_a,datatype):
    dataframe["Sex_Home"] = sex_h
    dataframe["Sex_Away"] = sex_a
    dataframe["Weight_Home"] = weight_h
    dataframe["Weight_Away"] = weight_a
    dataframe["Height_Home"] = height_h
    dataframe["Height_Away"] = height_a
    dataframe["Clock"] = clock
    dataframe["Left_hand_Home"] = left_hand_h
    dataframe["Left_hand_Away"] = left_hand_a
    dataframe["Age_Home"] = age_h
    dataframe["Age_Away"] = age_a
    dataframe["Year_Played_Home"] = year_played_h
    dataframe["Year_Played_Away"] = year_played_a
    dataframe["BMI_Home"] = dataframe["Weight_Home"] / ((dataframe["Height_Home"]/100)**2)
    dataframe["BMI_Away"] = dataframe["Weight_Away"] / ((dataframe["Height_Away"]/100)**2)
    dataframe["Total_Piezzo"] = dataframe["piezzo1"] + dataframe["piezzo2"] + dataframe["piezzo3"] + dataframe["piezzo4"]
    # dataframe.loc[(dataframe["BMI_Home"] < 20), "BMI_CAT_Home"] = "Thin"
    # dataframe.loc[((dataframe["BMI_Home"] >= 20) & (dataframe["BMI_Home"] < 25)), 'BMI_CAT_Home'] = "Normal"
    # dataframe.loc[(dataframe["BMI_Home"] >= 25), "BMI_CAT_Home"] = "Fat"
    # dataframe.loc[(dataframe["BMI_Away"] < 20), "BMI_CAT_Away"] = "Thin"
    # dataframe.loc[((dataframe["BMI_Away"] >= 20) & (dataframe["BMI_Away"] < 25)), 'BMI_CAT_Away'] = "Normal"
    # dataframe.loc[(dataframe["BMI_Away"] >= 25), "BMI_CAT_Away"] = "Fat"
    # dataframe.loc[(dataframe["Age_Home"] < 18), "AGE_CAT_Home"] = "Child"
    # dataframe.loc[((dataframe["Age_Home"] >= 18) & (dataframe["Age_Home"] < 30)), 'AGE_CAT_Home'] = "Young"
    # dataframe.loc[(dataframe["Age_Home"] >= 30), "AGE_CAT_Home"] = "Adult"
    # dataframe.loc[(dataframe["Year_Played_Home"] < 3), "Level_Home"] = "Beginner"
    # dataframe.loc[((dataframe["Year_Played_Home"] >= 3) & (dataframe["Year_Played_Home"] < 5)), 'Level_Home'] = "Mid"
    # dataframe.loc[(dataframe["Year_Played_Home"] >= 5), "Level_Home"] = "Professional"
    # dataframe.loc[(dataframe["Year_Played_Away"] < 3), "Level_Away"] = "Beginner"
    # dataframe.loc[((dataframe["Year_Played_Away"] >= 3) & (dataframe["Year_Played_Away"] < 5)), 'Level_Away'] = "Mid"
    # dataframe.loc[(dataframe["Year_Played_Away"] >= 5), "Level_Away"] = "Professional"
    return dataframe
def featureson(dataframe,sex_h,sex_a,weight_h,weight_a,height_h,height_a,clock,left_hand_h,left_hand_a,age_h,age_a,year_played_h,year_played_a,datatype):
    dataframe["Sex_Home"] = sex_h
    dataframe["Sex_Away"] = sex_a
    dataframe["Weight_Home"] = weight_h
    dataframe["Weight_Away"] = weight_a
    dataframe["Height_Home"] = height_h
    dataframe["Height_Away"] = height_a
    dataframe["Clock"] = clock
    dataframe["Left_hand_Home"] = left_hand_h
    dataframe["Left_hand_Away"] = left_hand_a
    dataframe["Age_Home"] = age_h
    dataframe["Age_Away"] = age_a
    dataframe["Year_Played_Home"] = year_played_h
    dataframe["Year_Played_Away"] = year_played_a
    dataframe["BMI_Home"] = dataframe["Weight_Home"] / ((dataframe["Height_Home"]/100)**2)
    dataframe["BMI_Away"] = dataframe["Weight_Away"] / ((dataframe["Height_Away"]/100)**2)
    dataframe["Total_Piezzo"] = dataframe["piezzo1"] + dataframe["piezzo2"] + dataframe["piezzo3"] + dataframe["piezzo4"]
    dataframe.loc[(dataframe["BMI_Home"] < 20), "BMI_CAT_Home"] = "Thin"
    dataframe.loc[((dataframe["BMI_Home"] >= 20) & (dataframe["BMI_Home"] < 25)), 'BMI_CAT_Home'] = "Normal"
    dataframe.loc[(dataframe["BMI_Home"] >= 25), "BMI_CAT_Home"] = "Fat"
    dataframe.loc[(dataframe["BMI_Away"] < 20), "BMI_CAT_Away"] = "Thin"
    dataframe.loc[((dataframe["BMI_Away"] >= 20) & (dataframe["BMI_Away"] < 25)), 'BMI_CAT_Away'] = "Normal"
    dataframe.loc[(dataframe["BMI_Away"] >= 25), "BMI_CAT_Away"] = "Fat"
    dataframe.loc[(dataframe["Age_Home"] < 18), "AGE_CAT_Home"] = "Child"
    dataframe.loc[((dataframe["Age_Home"] >= 18) & (dataframe["Age_Home"] < 30)), 'AGE_CAT_Home'] = "Young"
    dataframe.loc[(dataframe["Age_Home"] >= 30), "AGE_CAT_Home"] = "Adult"
    dataframe.loc[(dataframe["Age_Away"] < 18), "AGE_CAT_Away"] = "Child"
    dataframe.loc[((dataframe["Age_Away"] >= 18) & (dataframe["Age_Away"] < 30)), 'AGE_CAT_Away'] = "Young"
    dataframe.loc[(dataframe["Age_Away"] >= 30), "AGE_CAT_Away"] = "Adult"
    dataframe.loc[(dataframe["Year_Played_Home"] < 3), "Level_Home"] = "Beginner"
    dataframe.loc[((dataframe["Year_Played_Home"] >= 3) & (dataframe["Year_Played_Home"] < 5)), 'Level_Home'] = "Mid"
    dataframe.loc[(dataframe["Year_Played_Home"] >= 5), "Level_Home"] = "Professional"
    dataframe.loc[(dataframe["Year_Played_Away"] < 3), "Level_Away"] = "Beginner"
    dataframe.loc[((dataframe["Year_Played_Away"] >= 3) & (dataframe["Year_Played_Away"] < 5)), 'Level_Away'] = "Mid"
    dataframe.loc[(dataframe["Year_Played_Away"] >= 5), "Level_Away"] = "Professional"
    dataframe["Data_Type"] = datatype
    return dataframe

def clock_cycle(dataframe):
    dataframe['Clock'] = dataframe['Clock'].apply(lambda x: "morning" if x <= 12 else "evening")
    return dataframe

def sample(dataframe):
    dataframe = dataframe.sample(frac=1).reset_index(drop=True)
    return dataframe

# for iter,file in enumerate(files):
#     current_data = pd.read_csv(path+"/"+file , header=None,sep=" ")
#     headers = ["piezzo1", "piezzo2", "piezzo3", "piezzo4"]
#     current_data.columns = headers
#     current_data = feature(current_data, *listem[iter])
#     df = pd.concat([df,current_data])

def grab_col_names(dataframe, cat_th=10, car_th=20):

    # cat_cols, cat_but_car

    cat_cols = [col for col in dataframe.columns if dataframe[col].dtypes == "O"]

    num_but_cat = [col for col in dataframe.columns if dataframe[col].nunique() < cat_th and

                   dataframe[col].dtypes != "O"]

    cat_but_car = [col for col in dataframe.columns if dataframe[col].nunique() > car_th and

                   dataframe[col].dtypes == "O"]

    cat_cols = cat_cols + num_but_cat

    cat_cols = [col for col in cat_cols if col not in cat_but_car]



    # num_cols

    num_cols = [col for col in dataframe.columns if dataframe[col].dtypes != "O"]

    num_cols = [col for col in num_cols if col not in num_but_cat]



    return cat_cols, num_cols, cat_but_car

def one_hot_encoder(dataframe, categorical_cols, drop_first=False):

    dataframe = pd.get_dummies(dataframe, columns=categorical_cols, drop_first=drop_first)

    return dataframe

def scale():

    X_scaled = StandardScaler().fit_transform(df[num_cols])

    return X_scaled

def create_fin(dataframe,col):

    dataframe[col] = pd.DataFrame(X_scaled, columns=df[num_cols].columns)

    return dataframe

app = Flask(__name__)
model = pickle.load(open('rf_model5.pkl','rb')) #pickle'ı import et. Bu satır modeli çağırıyor.
@app.route('/')
def home():
    return render_template('template.html') #anasayfamız

##############################PREDICT KULLANILMIYOR BASLANGIC#####################################################################

@app.route('/predict', methods=['POST'])
def predict():
    int_features = [int(x) for x in request.form.values()]
    myListe = ["Female","Female",67, 60,165, 160, 19,"right_hand", "left_hand",20, 22,1,2]
    df = pd.read_csv('sensorValues.txt',sep=' ',header=None)
    headers = ['piezzo1','piezzo2','piezzo3','piezzo4']
    df.columns=headers
    # sayi = df['piezzo1'].values
    # print = ("asdasdasd--->>>>>", sayi)
    #print(df[['piezzo1','piezzo2','piezzo3','piezzo4']])
    print(df.values.tolist()[0][0],df.values.tolist()[0][1])
    df = feature(df,*myListe)
    df = clock_cycle(df)
    num_cols = [col for col in df.columns if df[col].dtypes != "O"]
    cat_cols = [col for col in df.columns if df[col].dtypes == "O"]
    df = one_hot_encoder(df, cat_cols, drop_first=False)
    #X_scaled = StandardScaler().fit_transform(df[num_cols])
    #print(X_scaled)
    #df[num_cols] = pd.DataFrame(X_scaled, columns=df[num_cols].columns)
    #df = create_fin(df,)
    #print(df)
    #df.head()
    #final_features = [np.array(int_features)]
    prediction = model.predict(df) #burada modele verileri sokuyoruz ve sonucu alıyoruz.
    print(prediction)
    #predicted_y = int(np.round(prediction,2)) #burasıda muhtemelen veriyi tam sayıya yuvarlama
    return render_template('template.html', prediction_text='Hello World : {0}'.format('hosbulduk'))

##############################PREDICT KULLANILMIYOR SON#####################################################################


@app.route("/members",methods=["GET"])
def members():
    args = request.args
    sexHomeTemp = args.get('sex_home')
    sexAwayTemp = args.get('sex_away')
    weightHomeTemp = args.get('weight_home')
    weightAwayTemp = args.get('weight_away')
    heightHomeTemp = args.get('height_home')
    heightAwayTemp = args.get('height_away')
    clockTemp = args.get('clock')
    handHomeTemp = args.get('hand_home')
    handAwayTemp = args.get('hand_away')
    ageHomeTemp = args.get('age_home')
    ageAwayTemp = args.get('age_away')
    experienceHomeTemp = args.get('experience_home')
    experienceAwayTemp = args.get('experience_away') 

    print(sexHomeTemp)
    print(sexAwayTemp)
    print(weightHomeTemp)
    print(weightAwayTemp)
    print(heightHomeTemp)
    print(heightAwayTemp)
    print(clockTemp)
    print(handHomeTemp)
    print(handAwayTemp)
    print(ageHomeTemp)
    print(ageAwayTemp)
    print(experienceHomeTemp)
    print(experienceAwayTemp)
    

    myListe = [sexHomeTemp,sexAwayTemp,int(weightHomeTemp), int(weightAwayTemp),int(heightHomeTemp), int(heightAwayTemp), int(clockTemp),handHomeTemp, handAwayTemp,int(ageHomeTemp), int(ageAwayTemp),int(experienceHomeTemp),int(experienceAwayTemp)]
    df = pd.read_csv('sensorValues.txt',sep=' ',header=None)
    headers = ['piezzo1','piezzo2','piezzo3','piezzo4']
    df.columns=headers

    df = feature99(df,*myListe)
    df.fillna(int(0),inplace = True)
    #df88 = feature2(df,*myListe)
    
    myListe2 = [sexHomeTemp,sexAwayTemp,int(weightHomeTemp), int(weightAwayTemp),int(heightHomeTemp), int(heightAwayTemp), int(clockTemp),handHomeTemp, handAwayTemp,int(ageHomeTemp), int(ageAwayTemp),int(experienceHomeTemp),int(experienceAwayTemp),"test"]
    df_anlık = pd.read_csv('sensorValues.txt',sep=' ',header=None)
    headers = ['piezzo1','piezzo2','piezzo3','piezzo4']
    df_anlık.columns=headers

    df_anlık = featureson(df_anlık,*myListe2)
    global tempDf3
    tempDf3 = pd.concat([tempDf3,df_anlık],ignore_index=True)
    #df = df.drop(columns = "Data_Type", axis = 1)
    print(df)
    print("-------------------------------------------------------")
    print(tempDf3)
    # df = clock_cycle(df)
    # num_cols = [col for col in df.columns if df[col].dtypes != "O"]
    # cat_cols = [col for col in df.columns if df[col].dtypes == "O"]
    # df = one_hot_encoder(df, cat_cols, drop_first=False)

    prediction = model.predict(df) #burada modele verileri sokuyoruz ve sonucu alıyoruz.
    print("buradayımmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmm1")
    #print(prediction)
    if prediction[0] == 0 :
        prediction = False
        print("buradayımmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmm2")
    elif prediction[0] == 1:
        prediction = True
        print("buradayımmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmm3")
    print("buradayımmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmm4")
    myObject = {"sensorValues": [df.values.tolist()[0][0],df.values.tolist()[0][1],df.values.tolist()[0][2],df.values.tolist()[0][3]] , "prediction" : prediction}
    print("buradayımmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmm5")
    #Buranın aşağısı eski kodlar
    # f = open("database.txt", "r") #bir önceki
    # myObject = f.readline() #bir önceki
    #myObject = '{"members": ["Member1", "Member2","Member3"]}'
    # print(myObject)
    # print("The type of object is: ", type(myObject))
    # myObject = json.loads(myObject)
    #print(myObject)
    #print("The type of object is: ", type(myObject))
    #print(myObject)
    #myObject = {"members": ["Member1", "Member2", "Member3"]}
    response = jsonify(myObject)
    # Enable Access-Control-Allow-Origin
    response.headers.add("Access-Control-Allow-Origin", "*")
    #print(response)
    #print("The type of object is: ", type(response))
    # f.close() #bir önceki
    return response

@app.route("/getabtest",methods=["GET"])
def getabtest():
    args = request.args
    colTemp = args.get('col') 
    global tempDf3,df2
    df_ful = pd.read_csv('datam.txt',sep='\t')

    tempDf3 = pd.concat([tempDf3,df_ful],ignore_index=True)
    resultText = ab_test(tempDf3,col=colTemp,target="Total_Piezzo")
    
    myObject = {"abTestResult" : resultText, "abTestColName" : colTemp}
    response = jsonify(myObject)
    response.headers.add("Access-Control-Allow-Origin", "*")
    # print("-------------------------------------------" + resultText + "-------------------------------------")
    # print("-------------------------------------------" + resultText + "-------------------------------------")
    # print("-------------------------------------------" + resultText + "-------------------------------------")
    # print("-------------------------------------------" + resultText + "-------------------------------------")
    return response
    

if __name__ == "__main__":
    app.run(host="0.0.0.0", port="5000",debug=True)

#     listem=[["Female","Female",67, 60,165, 160, 19,"right_hand", "left_hand",20, 22,1,2],
# ["Male","Female", 85,53, 190,160, 22, "right_hand","right_hand",21, 21,7,0],
# ["Female","Male", 62,69, 178,171, 20, "right_hand","left_hand", 23,19,1,3],
# ["Male","Male",83, 78, 185,181, 11, "right_hand","left_hand",21, 22,3,4],
# ["Male","Female", 80,59, 187,173, 21, "left_hand","right_hand",22, 23,4,1],
# ["Female","Male", 66, 78, 169,178, 9, "right_hand","left_hand",27, 24,5,2],
# ["Male","Male", 70,80, 170,180, 10, "left_hand","right_hand",20, 25,2,3],
# ["Male","Female",83,72, 176,179, 23, "right_hand","right_hand", 39,25,4,0],
# ["Male","Male", 80,83, 187,186, 21, "left_hand","left_hand", 23,22,4,2],
# ["Female","Male", 74,92, 172,179, 20, "right_hand","right_hand", 26,24,6,2]]

# def feature(dataframe,sex_h,sex_a,weight_h,weight_a,height_h,height_a,clock,left_hand_h,left_hand_a,age_h,age_a,year_played_h,year_played_a):
#     dataframe["Sex_Home"] = sex_h
#     dataframe["Sex_Away"] = sex_a
#     dataframe["Weight_Home"] = weight_h
#     dataframe["Weight_Away"] = weight_a
#     dataframe["Height_Home"] = height_h
#     dataframe["Height_Away"] = height_a
#     dataframe["Clock"] = clock
#     dataframe["Left_hand_Home"] = left_hand_h
#     dataframe["Left_hand_Away"] = left_hand_a
#     dataframe["Age_Home"] = age_h
#     dataframe["Age_Away"] = age_a
#     dataframe["Year_Played_Home"] = year_played_h
#     dataframe["Year_Played_Away"] = year_played_a
#     dataframe["BMI_Home"] = dataframe["Weight_Home"] / ((dataframe["Height_Home"]/100)**2)
#     dataframe["BMI_Away"] = dataframe["Weight_Away"] / ((dataframe["Height_Away"]/100)**2)
#     dataframe["Total_Piezzo"] = dataframe["piezzo1"] + dataframe["piezzo2"] + dataframe["piezzo3"] + dataframe["piezzo4"]
#     # dataframe.loc[(dataframe["BMI_Home"] < 20), "BMI_CAT_Home"] = "Thin"
#     # dataframe.loc[((dataframe["BMI_Home"] >= 20) & (dataframe["BMI_Home"] < 25)), 'BMI_CAT_Home'] = "Normal"
#     # dataframe.loc[(dataframe["BMI_Home"] >= 25), "BMI_CAT_Home"] = "Fat"
#     # dataframe.loc[(dataframe["BMI_Away"] < 20), "BMI_CAT_Away"] = "Thin"
#     # dataframe.loc[((dataframe["BMI_Away"] >= 20) & (dataframe["BMI_Away"] < 25)), 'BMI_CAT_Away'] = "Normal"
#     # dataframe.loc[(dataframe["BMI_Away"] >= 25), "BMI_CAT_Away"] = "Fat"
#     # dataframe.loc[(dataframe["Age_Home"] < 18), "AGE_CAT_Home"] = "Child"
#     # dataframe.loc[((dataframe["Age_Home"] >= 18) & (dataframe["Age_Home"] < 30)), 'AGE_CAT_Home'] = "Young"
#     # dataframe.loc[(dataframe["Age_Home"] >= 30), "AGE_CAT_Home"] = "Adult"
#     # dataframe.loc[(dataframe["Year_Played_Home"] < 3), "Level_Home"] = "Beginner"
#     # dataframe.loc[((dataframe["Year_Played_Home"] >= 3) & (dataframe["Year_Played_Home"] < 5)), 'Level_Home'] = "Mid"
#     # dataframe.loc[(dataframe["Year_Played_Home"] >= 5), "Level_Home"] = "Professional"
#     # dataframe.loc[(dataframe["Year_Played_Away"] < 3), "Level_Away"] = "Beginner"
#     # dataframe.loc[((dataframe["Year_Played_Away"] >= 3) & (dataframe["Year_Played_Away"] < 5)), 'Level_Away'] = "Mid"
#     # dataframe.loc[(dataframe["Year_Played_Away"] >= 5), "Level_Away"] = "Professional"
#     return dataframe
# #sex_h,sex_a,weight_h,weight_a,height_h,height_a,clock,left_hand_h,left_hand_a,age_h,age_a,year_played_h,year_played_a