from flask import Flask, request, jsonify
import json
from tensorflow import keras
import numpy as np

app = Flask(__name__)

minMCHC = 18.07
maxMCHC = 168.0
minHCT = 7.21
maxHCT = 67.4
minHGB = 2.99
maxHGB = 22.45
minMCV = 49.93
maxMCV = 125.3
minPLT = 13
maxPLT = 1770
minFERRITTE = 0.5
maxFERRITTE = 27332.0
minB12 = 30.0
maxB12 = 33880.0
minFOLATE = 0.55
maxFOLATE = 50.25

minTB = 0.4
maxTB = 75.0
minSGPT = 10
maxSGPT = 2000
minSGOT = 10
maxSGOT = 4929
minALB = 0.9
maxALB = 5.5

minWBC = 0.57
maxWBC = 246.7
minRBC = 0.87
maxRBC = 7.5
minRDW = 9.72
maxRDW = 39.4


liver_model = keras.models.load_model('models/liver.h5')
CBCA_model = keras.models.load_model('models/CBC_Advance.h5')
CBC_model = keras.models.load_model('models/cbc.h5')


@app.route('/liver_pred', methods=['POST'])
def liver_pred():

    input_data = request.get_json()

    TB = (input_data['TotalBilirubin'] - minTB) / (maxTB - minTB)
    Sgpt = (input_data['SgptAlamineAminotransferase'] - minSGPT) / (maxSGPT - minSGPT)
    Sgot = (input_data['SgotAspartateAminotransferase'] - minSGOT) / (maxSGOT - minSGOT)
    ALB = (input_data['ALBAlbumin'] - minALB) / (maxALB - minALB)
    gender = input_data['gender_dummy']

    input_list = [TB, Sgpt, Sgot, ALB, gender]

    prediction = liver_model.predict([input_list])

    class_labels = np.argmax(prediction, axis=1)

    if class_labels[0] == 0:
        return "Nothing"
    elif class_labels[0] == 1:
        return "Virus C"
    elif class_labels[0] == 2:
        return "Gallbladder"
    elif class_labels[0] == 3:
        return "Virus A"
    elif class_labels[0] == 4:
        return "Fatty Liver"


@app.route('/CBCA_pred', methods=['POST'])
def CBCA_pred():

    input_data = request.get_json()
    
    MCHC = (input_data['MCHC'] - minMCHC) / (maxMCHC - minMCHC)
    HCT = (input_data['HCT'] - minHCT) / (maxHCT - minHCT)
    HGB = (input_data['HGB'] - minHGB) / (maxHGB - minHGB)
    MCV = (input_data['MCV'] - minMCV) / (maxMCV - minMCV)
    PLT = (input_data['PLT'] - minPLT) / (maxPLT - minPLT)
    FERRITTE = (input_data['FERRITTE'] - minFERRITTE) / (maxFERRITTE - minFERRITTE)
    B12 = (input_data['B12'] - minB12) / (maxB12 - minB12)
    FOLATE = (input_data['FOLATE'] - minFOLATE) / (maxFOLATE - minFOLATE)
    GENDER = input_data['GENDER']

    # X=np.asarray(df[['MCHC','HCT','HGB','MCV','PLT','FERRITTE','B12','FOLATE','GENDER']].values.tolist())

    input_list = [MCHC, HCT, HGB, MCV, PLT, FERRITTE, B12, FOLATE, GENDER]

    prediction = CBCA_model.predict([input_list])

    class_labels = np.argmax(prediction, axis=1)

    if class_labels[0] == 0:
        return "Nothing"
    elif class_labels[0] == 1:
        return "HGB anemia"
    elif class_labels[0] == 2:
        return "iron anemia"
    elif class_labels[0] == 3:
        return "Folate anemia"
    elif class_labels[0] == 4:
        return "B12 anemia"


@app.route('/CBC_pred', methods=['POST'])
def CBC_pred():
    input_data = request.get_json()

    WBC = (input_data['WBC'] - minWBC) / (maxWBC - minWBC)
    RBC = (input_data['RBC'] - minRBC) / (maxRBC - minRBC)
    HGB = (input_data['HGB'] - minHGB) / (maxHGB - minHGB)
    MCV = (input_data['MCV'] - minMCV) / (maxMCV - minMCV)
    RDW = (input_data['RDW'] - minRDW) / (maxRDW - minRDW)
    PLT = (input_data['PLT'] - minPLT) / (maxPLT - minPLT)
    GENDER = input_data['GENDER']

    # X=np.asarray(df[['MCHC','HCT','HGB','MCV','PLT','FERRITTE','B12','FOLATE','GENDER']].values.tolist())

    input_list = [WBC, RBC, HGB, MCV, RDW, PLT, GENDER]
    prediction = CBC_model.predict([input_list])

    class_labels = np.argmax(prediction, axis=1)

    if class_labels[0] == 0:
        return "Nothing"
    elif class_labels[0] == 1:
        return "hemolytic"
    elif class_labels[0] == 2:
        return "macrocytic"
    elif class_labels[0] == 3:
        return "penectopenia"
    elif class_labels[0] == 4:
        return "microcytic hypochromic"


if __name__ == '__main__':
    app.run()

