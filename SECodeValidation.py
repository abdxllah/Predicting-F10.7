
#This is a secondary script for the code that lets you do cross validation to see if results are consistent
#across different executions of the model- it is only commented with the changes that are significant from the primary 
#script

#THIS IS NOT THE MAIN CODE - Just an extra evaluation metric

import numpy as numpyImport
import pandas as pandasImport
import tensorflow as tensorFlowImport
import matplotlib.pyplot as pyPlotImport
import seaborn as correlation
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error, r2_score, explained_variance_score, median_absolute_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import LearningRateScheduler, EarlyStopping
from scipy.stats import probplot




datasetPrimary = pandasImport.read_csv('maindata.csv') #load the dataset
datasetPrimary['Datetime'] = pandasImport.to_datetime(datasetPrimary['Datetime'])
datasetPrimary = datasetPrimary.set_index('Datetime')

datasetPrimary = datasetPrimary.drop(columns=['F3']) #dropping the f3 column
datasetPrimary = datasetPrimary.drop(columns=['F30']) # dropping the f30 column

#removing any 0 values in the flux density + ssn
datasetPrimary = datasetPrimary[(datasetPrimary['F10.7'] != 0)]
datasetPrimary = datasetPrimary[(datasetPrimary['F15'] != 0)]
datasetPrimary = datasetPrimary[(datasetPrimary['F8'] != 0)]
datasetPrimary = datasetPrimary[(datasetPrimary['SSN'] != 0)]

for i in range(1, 5):
    datasetPrimary[f'F10.7_Lag{i}'] = datasetPrimary['F10.7'].shift(i)
    datasetPrimary[f'SSN_Lag{i}'] = datasetPrimary['SSN'].shift(i)
datasetPrimary = datasetPrimary.dropna()
for i in range(1, 2): 
    datasetPrimary[f'F15_Lag{i}'] = datasetPrimary['F15'].shift(i)
datasetPrimary = datasetPrimary.dropna()
datasetPrimary.to_csv('check.csv') 

scaler = MinMaxScaler() 
datasetScaled = scaler.fit_transform(datasetPrimary)

#sequence length is defined earlier here
sequenceLength = 30  

#create sequences becomes a function to be used in the loop
def sequenceCreation(data, sequenceLength):
    X, y = [], []
    for i in range(len(data) - sequenceLength):
        X.append(data[i:i+sequenceLength, :])
        y.append(data[i+sequenceLength, 0])
    return numpyImport.array(X), numpyImport.array(y)

# Time series cross-validation of 5 splits
timeCrossValidation = TimeSeriesSplit(n_splits=5)  

for train_index, test_index in timeCrossValidation.split(datasetScaled):
    train_data, test_data = datasetScaled[train_index], datasetScaled[test_index]

    xTraining, yTraining = sequenceCreation(train_data, sequenceLength)
    xTesting, yTesting = sequenceCreation(test_data, sequenceLength)

    model = Sequential()
    model.add(LSTM(units=50, activation='relu', input_shape=(xTraining.shape[1], xTraining.shape[2])))
    model.add(Dropout(0.2))
    model.add(Dense(units=1))
    model.compile(optimizer='adam', loss='mean_squared_error')
    epochs = 50
    batches = 64
    history = model.fit(xTraining, yTraining, epochs=epochs, batch_size=batches, verbose=0)
    predictionsFull = model.predict(xTesting)
    predictedNormalised = scaler.inverse_transform(numpyImport.concatenate((xTesting[:, -1, 1:], predictionsFull), axis=1))[:, -1]
    testingNormalised = scaler.inverse_transform(numpyImport.concatenate((xTesting[:, -1, 1:], yTesting.reshape(-1, 1)), axis=1))[:, -1]
    
    meanAbsoluteErrorPredictions = mean_absolute_error(testingNormalised, predictedNormalised)
    rootMeanSquaredErrorPredictions = numpyImport.sqrt(mean_squared_error(testingNormalised, predictedNormalised))
    meanAbsolutePercentageErrorPredictions = mean_absolute_percentage_error(testingNormalised, predictedNormalised)
    medianAbsoluteErrorPredictions = median_absolute_error(testingNormalised, predictedNormalised)
    r2Predicitons = r2_score(testingNormalised, predictedNormalised)
    explainedVariancePredictions = explained_variance_score(testingNormalised, predictedNormalised)

    print(f'Mean Absolute Error of the predictions is {meanAbsoluteErrorPredictions:.2f}')
    print(f'Root Mean Squared Error of the predictions is {rootMeanSquaredErrorPredictions:.2f}')
    print(f'Mean absolute percentage error of the predictions is {meanAbsolutePercentageErrorPredictions:.2f}')
    print(f'Median Absolute Error of the predictions is  {medianAbsoluteErrorPredictions:.2f}')
    print(f'R-squared of the predictions is  {r2Predicitons:.2f}')
    print(f'Explained Variance of the predictions is  {explainedVariancePredictions:.2f}')
    print(f'Median Absolute Error of the predictions is  {medianAbsoluteErrorPredictions:.2f}')

 