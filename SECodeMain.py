#The following code provides a prediction for the next 28 days for F10.7cm flux density
#It does this by using historical F10.7cm data, alongside historical Sunspot number data 
#and other flux densities F15cm and F8cm. Other drivers have been attempted, but only negatively
#detriment the model.
#It makes use of an LSTM (Long Short-Term Memory) model that are well suited to time series 
#prediction tasks due to: its handling of temporal dependencies and its ability to 
#capture patterns in time.

#To run this code you will need to pip install some packages
#The line below shows what to run in the terminal to be able to run this code

# pip install numpy pandas tensorflow scikit-learn matplotlib seaborn scipy

#you can then use terminal or command line to run the code using the command whilst being in the folder (having this folder as the current working directory):
#python3 SECodeMain.py 
#Ensure that all other files required are in the same folder (csvs)

#this was programmed on mac but can also run on linux command line like this and windows
#pc's with python installed

import numpy as numpyImport
import pandas as pandasImport
import tensorflow as tensorFlowImport
import matplotlib.pyplot as pyPlotImport
import seaborn as correlation
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error, r2_score, explained_variance_score, median_absolute_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import LearningRateScheduler, EarlyStopping
from scipy.stats import probplot

#Drivers used in this code are influenced by 'What to Do When the F10.7 Goes Out?' 
# - Sean Elvidge, David R. Themens, Matthew K. Brown, Elizabeth Donegan-Lawley. 
# https://agupubs.onlinelibrary.wiley.com/doi/10.1029/2022SW003392


#Data used for this code is from three CSV files. One of these contains F10.7cm historical
#data alongside Sunspot data and other flux densitites (maindata.csv) and the other two
#CSV's are kpapdataset.csv(which contains F10.7cm, sunspot number and kp and ap data) and
# xrayflux.csv (which contains F10.7cm, sunspot number, F15cm and xray flux data). 

#These CSV's were made predominantly from the data provided (for F10.7cm historical as
# well as sunspot data and kp and ap data) and using code like below to 
#combine datasets over their common dates (where data had a time as well, 20:00:00 was used
# corresponding to the time in the historical F10.7cm data):

#datasetPrimary = pandasImport.read_csv('first.csv, parse_dates = ['Datetime'])
#datasetSecondary = pandasImport.read_csv('second.csv, parse_dates = ['Datetime'])
#datasetPrimary ['Date'] = datasetPrimary[Datetime].dt.date
#datasetSecondary = pandasImport.to_datetime(datasetSecondary['Datetime']).dt.date
#mergedDataset = pandasImport.merge(datasetPrimary, datasetSecondary, on= "Date", how = "inner")
#mergedDataset = mergedDataset.drop(columns= ['Date', Datetime_y])
#mergedDataset.to_csv(datasetMerged.csv, index = False)

#Data for other radio flux densities was obtained by National Astronomical Observatory of Japan -
# https://solar.nro.nao.ac.jp/norp/
#Data for Xray was obtained through NOAA'S Goes Satellite - 
# https://www.ngdc.noaa.gov/stp/satellite/goes-r.html


#When starting this project it was important to decide what drivers would help produce the 
#best model. Historical F10.7cm was important, alongside sunspot number (which are strongly
# correlated) and xray/solar flare data was also originally deemed important. To decide which
# drivers to take forward correlation matrix's were plotted.

#correlation matrix for maindata.csv
datasetPrimary = pandasImport.read_csv('maindata.csv') #load the dataset
datasetPrimary['Datetime'] = pandasImport.to_datetime(datasetPrimary['Datetime'])
datasetPrimary = datasetPrimary.set_index('Datetime')
collumnsCorrelated = ['F10.7', 'SSN', 'F3', 'F8', 'F15', 'F30']
correlationData = datasetPrimary[collumnsCorrelated] # creating a subset dataframe
correlationMatrix = correlationData.corr()#calculating the correlation matrix
pyPlotImport.figure(figsize=(10, 8)) # visualisation
correlation.heatmap(correlationMatrix, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
pyPlotImport.title('Correlation Matrix of F10.7, SSN, F3 F8, F15 and F30')
pyPlotImport.show() 

#correlation matrix for kpap data 
datasetKpAp = pandasImport.read_csv('kpapdataset.csv')  
datasetKpAp['Datetime'] = pandasImport.to_datetime(datasetKpAp['Datetime'])
datasetKpAp = datasetKpAp.set_index('Datetime')
collumnsCorrelatedTwo = ['F10.7', 'Kp', 'Ap']
correlationData = datasetKpAp[collumnsCorrelatedTwo]
correlationMatrix = correlationData.corr()
pyPlotImport.figure(figsize=(10, 8))
correlation.heatmap(correlationMatrix, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
pyPlotImport.title('Correlation Matrix of F10.7, Kp and Ap')
pyPlotImport.show()

#correlation matrix for xray flux data 
datasetXray = pandasImport.read_csv('xrayflux.csv')  
datasetXray['Datetime'] = pandasImport.to_datetime(datasetXray['Datetime'])
datasetXray = datasetXray.set_index('Datetime')
columns_to_correlate3 = ['F10.7', 'xray_flux']
correlationData = datasetXray[columns_to_correlate3]
correlationMatrix = correlationData.corr()
pyPlotImport.figure(figsize=(10, 8))
correlation.heatmap(correlationMatrix, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
pyPlotImport.title('Correlation Matrix of F10.7 and Xray Flux')
pyPlotImport.show()

#It was decided to use the drivers in the main data set except F3cm and F30cm as they led to negative
#performance in the model. Also in the paper referenced above
# the use of F15 alongside F8 led to good performance when 
# trying to calculate F10.7cm without historical data  - Now we process the data intially

datasetPrimary = datasetPrimary.drop(columns=['F3']) #dropping the f3 column
datasetPrimary = datasetPrimary.drop(columns=['F30']) # dropping the f30 column

#removing any 0 values in the flux density + ssn
datasetPrimary = datasetPrimary[(datasetPrimary['F10.7'] != 0)]
datasetPrimary = datasetPrimary[(datasetPrimary['F15'] != 0)]
datasetPrimary = datasetPrimary[(datasetPrimary['F8'] != 0)]
datasetPrimary = datasetPrimary[(datasetPrimary['SSN'] != 0)]

#since we are doing time based prediction - lag features are beneficial - so now 
#we create these.
for i in range(1, 5): # 5 led to good performance
    datasetPrimary[f'F10.7_Lag{i}'] = datasetPrimary['F10.7'].shift(i)
    datasetPrimary[f'SSN_Lag{i}'] = datasetPrimary['SSN'].shift(i)
datasetPrimary = datasetPrimary.dropna()#we drop any missing values from the dataset here too
for i in range(1, 2): #no lagging for f8 as causes overfitting
    datasetPrimary[f'F15_Lag{i}'] = datasetPrimary['F15'].shift(i)
datasetPrimary = datasetPrimary.dropna()
datasetPrimary.to_csv('check.csv') #we can produce a csv here to ensure that our data is correct

# Now we start setting up the model
scaler = MinMaxScaler() # we scale the date using minmax so that training is more stable
# and convergence occurs quicker
datasetScaled = scaler.fit_transform(datasetPrimary) #scaling the data
#splitting the data into testing and training
dataSplit = int(0.8 * len(datasetPrimary)) 
trainingData = datasetScaled[:dataSplit]
testingData = datasetScaled[dataSplit:]
# convert to sequences to allow for LSTM to work on data
def sequenceCreation(data, sequenceLen):
    x, y = [], []
    for i in range(len(data) - sequenceLen):
        x.append(data[i:i+sequenceLen, :])
        y.append(data[i+sequenceLen, 0])
    return numpyImport.array(x), numpyImport.array(y)
sequenceLen = 30 
xTraining, yTraining = sequenceCreation(trainingData, sequenceLen)
xTesting, yTesting = sequenceCreation(testingData, sequenceLen)
# building LSTM model and setting its parameters / defining the architecture of the model
model = Sequential()
model.add(LSTM(units=50, activation='relu', input_shape=(xTraining.shape[1], xTraining.shape[2])))
model.add(Dropout(0.2))
model.add(Dense(units=1))
model.compile(optimizer='adam', loss='mean_squared_error')
# we defined a learning rate to optimise the speed of the model and avoid overshooting
def learningRate(epoch):
    initialLR = 0.001
    decay = 0.9
    return initialLR * decay**(epoch // 10)
# create a scheduler to dynamically adjust over time
learningRateScheduler = LearningRateScheduler(learningRate)
epochs = 50 #set our number of epochs and batches
batches = 64  
earlyStopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)#early stopping
# to reduce the chance of overfitting
history = model.fit(xTraining, yTraining, epochs=epochs, batch_size=batches,
                    validation_data=(xTesting, yTesting), callbacks=[earlyStopping, learningRateScheduler])

# Plotting the training of the model
pyPlotImport.plot(history.history['loss'], label='Training')
pyPlotImport.plot(history.history['val_loss'], label='Validation')
pyPlotImport.legend()
pyPlotImport.show()

# evaluating the model and inverse transforming the data
predictionsFull = model.predict(xTesting)
predictedNormalised = scaler.inverse_transform(numpyImport.concatenate((xTesting[:, -1, 1:], predictionsFull), axis=1))[:, -1]
testingNormalised = scaler.inverse_transform(numpyImport.concatenate((xTesting[:, -1, 1:], yTesting.reshape(-1, 1)), axis=1))[:, -1]

# printing the predicted values for the next 28 days
predictions = predictedNormalised[-28:]
print("Predicted F10.7 values for the next 28 days:")
print(predictions)

#these are the actual f10.7 values after my data ends
actualValues = numpyImport.array([161.5, 157.7, 153.8, 155.1, 156.1, 155.3, 161.8, 156.8, 165.2, 163.9,
                          157.5, 154.3, 148.4, 147.4, 143.8, 143.1, 136.4, 134.3, 127.7, 124.6,
                          121.4, 117.6, 120.8, 119.8, 124.4, 124.9, 125.9,126.3])

#here we sent a benchmark - we want to be better than just taking the average of the
#last 28 days and using that 
latest28F107 = datasetPrimary['F10.7'].tail(28).values
averageLatestF107 = numpyImport.mean(latest28F107) # calculate the average for know 28 values
averageLatestF107Array = numpyImport.full_like(testingNormalised, averageLatestF107)
#do the evaluation metrics that will be done on the predicted data on this
meanAbsoluteErrorAverage = mean_absolute_error(testingNormalised, averageLatestF107Array)
rootMeanSquaredErrorAverage = numpyImport.sqrt(mean_squared_error(testingNormalised, averageLatestF107Array))
print(f'Average F10.7cm for the last 28 days is {averageLatestF107:.2f}')
print(f'Mean Absolute Error if we use the average of the last 28 days is {meanAbsoluteErrorAverage:.2f}')
print(f'Root Mean Squared Error if we use the average of the last 28 days is {rootMeanSquaredErrorAverage:.2f}')

#now we produce evaluation metrics for the predictings
meanAbsoluteErrorPredictions = mean_absolute_error(testingNormalised, predictedNormalised)
#average absolute difference between actual & predicted values
rootMeanSquaredErrorPredictions = numpyImport.sqrt(mean_squared_error(testingNormalised, predictedNormalised))
#similar to mean absolute but penalises larger errors heavier
meanAbsolutePercentageErrorPredictions = mean_absolute_percentage_error(testingNormalised, predictedNormalised)
#expresses errors as a percentage of actual value
medianAbsoluteErrorPredictions = median_absolute_error(testingNormalised, predictedNormalised)
#similar to mean absolute error but less sensitive to outliers
r2Predicitons = r2_score(testingNormalised, predictedNormalised)
#shows how much of the variation of depedent variable is predictable from independent variable 
explainedVariancePredictions = explained_variance_score(testingNormalised, predictedNormalised)


print(f'Mean Absolute Error of the predictions is {meanAbsoluteErrorPredictions:.2f}')
print(f'Root Mean Squared Error of the predictions is {rootMeanSquaredErrorPredictions:.2f}')
print(f'Mean absolute percentage error of the predictions is {meanAbsolutePercentageErrorPredictions:.2f}')
print(f'Median Absolute Error of the predictions is  {medianAbsoluteErrorPredictions:.2f}')
print(f'R-squared of the predictions is  {r2Predicitons:.2f}')
print(f'Explained Variance of the predictions is  {explainedVariancePredictions:.2f}')
print(f'Median Absolute Error of the predictions is  {medianAbsoluteErrorPredictions:.2f}')

# we also perform residual analysis to evaluate the model
residuals = testingNormalised - predictedNormalised

# we plot the residuals
pyPlotImport.figure(figsize=(12, 6))
pyPlotImport.plot(residuals, marker='o', linestyle='None', color='r')
pyPlotImport.axhline(y=0, color='k', linestyle='--', label='Zero Residual line')
pyPlotImport.title('Residual analysis')
pyPlotImport.xlabel('Sample index')
pyPlotImport.ylabel('Residuals')
pyPlotImport.legend()
pyPlotImport.show()

#we also produce a histogram to evalute performance
pyPlotImport.figure(figsize=(10, 6))
pyPlotImport.hist(residuals, bins=30, edgecolor='black')
pyPlotImport.title('Histogram of residuals')
pyPlotImport.xlabel('Residuals')
pyPlotImport.ylabel('Frequency')
pyPlotImport.show()

# a Q-Q Plot of residuals is also created to perform evaluation
pyPlotImport.figure(figsize=(8, 8))
probplot(residuals, plot=pyPlotImport)
pyPlotImport.title('Q-Q Plot of Residuals')
pyPlotImport.show()
print(residuals)

# We now print the actual values for the next 28 days
print("Actual F10.7 values for the next 28 days:")
print(actualValues)
# a graphcomparing actual and predicted values
pyPlotImport.figure(figsize=(12, 6))
pyPlotImport.plot(predictions, label='Predicted', marker='o')
pyPlotImport.plot(actualValues, label='Actual', marker='o')
pyPlotImport.title('Comparison of Actual and Predicted F10.7 Values for the Next 28 Days')
pyPlotImport.xlabel('Day')
pyPlotImport.ylabel('F10.7')
pyPlotImport.legend()
pyPlotImport.show()

# Save the model
model.save('f107model.h5')

