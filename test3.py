import pandas as pd
import numpy as np
from sklearn import metrics
from sklearn import preprocessing as pp
from sklearn.model_selection import train_test_split 
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestRegressor

def add_noise(series, noise_level):
    return series * (1 + noise_level * np.random.randn(len(series)))

#Function to perform target encoding
def target_encode(trn_series,tst_series,target):
    min_samples_leaf=1 
    smoothing=1,
    noise_level=0
    temp = pd.concat([trn_series, target], axis=1)
    # Compute target mean 
    averages = temp.groupby(by=trn_series.name)[target.name].agg(["mean", "count"])
    # Compute smoothing
    smoothing = 1 / (1 + np.exp(-(averages["count"] - min_samples_leaf) / smoothing))
    # Apply average function to all target data
    prior = target.mean()
    # The bigger the count the less full_avg is taken into account
    averages[target.name] = prior * (1 - smoothing) + averages["mean"] * smoothing
    averages.drop(["mean", "count"], axis=1, inplace=True)
    # Apply aver
    assert len(trn_series) == len(target)
    assert trn_series.name == tst_series.name
    temp = pd.concat([trn_series, target], axis=1)
    # Compute target mean 
    averages = temp.groupby(by=trn_series.name)[target.name].agg(["mean", "count"])
    # Compute smoothing
    smoothing = 1 / (1 + np.exp(-(averages["count"] - min_samples_leaf) / smoothing))
    # Apply average function to all target data
    prior = target.mean()
    # The bigger the count the less full_avg is taken into account
    averages[target.name] = prior * (1 - smoothing) + averages["mean"] * smoothing
    averages.drop(["mean", "count"], axis=1, inplace=True)
    # Apply averages to trn and tst series
    ft_trn_series = pd.merge(
        trn_series.to_frame(trn_series.name),
        averages.reset_index().rename(columns={'index': target.name, target.name: 'average'}),
        on=trn_series.name,
        how='left')['average'].rename(trn_series.name + '_mean').fillna(prior)
    # pd.merge does not keep the index so restore it
    ft_trn_series.index = trn_series.index 
    ft_tst_series = pd.merge(
        tst_series.to_frame(tst_series.name),
        averages.reset_index().rename(columns={'index': target.name, target.name: 'average'}),
        on=tst_series.name,
        how='left')['average'].rename(trn_series.name + '_mean').fillna(prior)
    # pd.merge does not keep the index so restore it
    ft_tst_series.index = tst_series.index
    return add_noise(ft_trn_series, noise_level), add_noise(ft_tst_series, noise_level)

#Read csv file into dataframe Maindf
Maindf = pd.read_csv('D:/Trinity/Machine learning/Kaggle/tcd ml 2019-20 income prediction training (with labels).csv', na_values = {
    'Year of Record': ["#NA"],
    'Gender': ["#NA","0","unknown"],
    'Age': ["#NA"],
    'Profession':["#NA"],
    'University Degree' : ["0","#NA"],
    'Hair Color': ["#NA","0","Unknown"]
 } )

Maindf.to_csv('testcleaned-data.csv')

#reading the test data
Testdf = pd.read_csv('D:/Trinity/Machine learning/Kaggle/tcd ml 2019-20 income prediction test (without labels).csv', na_values = {
    'Year of Record': ["#NA"],
    'Gender': ["#NA","0","unknown"],
    'Age': ["#NA"],
    'Profession':["#NA"],
    'University Degree' : ["0","#NA"],
    'Hair Color': ["#NA","0","Unknown"]
 } )

#Testdf.to_csv('testcleaned-testdata.csv')

#Identifying and filling missing values for the features
Maindf['Year of Record'].fillna(Maindf['Year of Record'].interpolate(method='slinear'), inplace=True)
Maindf['Gender'].fillna('other', inplace=True)
Maindf['Age'].fillna(Maindf['Age'].interpolate(method='slinear'), inplace=True)
Maindf['Gender'].fillna(method="ffill", inplace=True)
Maindf['Profession'].fillna(method="ffill", inplace=True)
Maindf['University Degree'].fillna(method="ffill", inplace=True)
Maindf['Hair Color'].fillna(method="ffill", inplace=True)

#Maindf.to_csv('testfilled-data.csv')

#Identifying and filling missing values for the features in test data
Testdf['Year of Record'].fillna(Testdf['Year of Record'].interpolate(method='slinear'), inplace=True)
Testdf['Gender'].fillna('other', inplace=True)
Testdf['Age'].fillna(Testdf['Age'].interpolate(method='slinear'), inplace=True)
Testdf['Gender'].fillna(method="ffill", inplace=True)
Testdf['Profession'].fillna(method="ffill", inplace=True)
Testdf['University Degree'].fillna(method="ffill", inplace=True)
Testdf['Hair Color'].fillna(method="ffill", inplace=True)

#Testdf.to_csv('testfilled-testdata.csv')

# #SCaling features - Age and Year of Record
age_scaler = pp.StandardScaler()
Maindf['Age'] = age_scaler.fit_transform(Maindf['Age'].values.reshape(-1, 1))

yor_scaler = pp.StandardScaler()
Maindf['Year of Record'] = yor_scaler.fit_transform(Maindf['Year of Record'].values.reshape(-1, 1))

# Maindf.to_csv('testscaled-data.csv')

#SCaling features - Age and Year of Record of test data
Testdf['Age'] = age_scaler.transform(Testdf['Age'].values.reshape(-1, 1))
Testdf['Year of Record'] = yor_scaler.transform(Testdf['Year of Record'].values.reshape(-1, 1))

# Testdf.to_csv('testscaled-testdata.csv')

#Label encoding for feature Gender for training data
le_Gender = pp.LabelEncoder() 
Maindf['Gender'] = le_Gender.fit_transform(Maindf['Gender']) 
Maindf['Gender'].unique()


# #Label encoding for feature University Degree for training data
le_UniDeg = pp.LabelEncoder()
Maindf['University Degree'] = le_UniDeg.fit_transform(Maindf['University Degree']) 
Maindf['University Degree'].unique()


# #Label encoding for feature Hair Color for training data
le_HairClr = pp.LabelEncoder()
Maindf['Hair Color'] = le_HairClr.fit_transform(Maindf['Hair Color']) 
Maindf['Hair Color'].unique()


#Removed the feature Income in EUR from data frame and equated to Y
Y = Maindf['Income in EUR']
Y = Y.abs()
#print(Y)
Maindf = Maindf.drop(columns=['Income in EUR'])
#print(Maindf.head())

# #Label encoding for feature Profession of training data
professionList = Maindf['Profession'].unique()
professionReplaced = Maindf.groupby('Profession').count()
professionReplaced = professionReplaced[professionReplaced['Age'] < 3].index
Maindf['Profession'].replace(professionReplaced, 'other profession', inplace=True)

le_Prof = pp.LabelEncoder()
Maindf['Profession'] = le_Prof.fit_transform(Maindf['Profession']) 
Maindf['Profession'].unique()


# #Label encoding for feature Country of training data
countryList = Maindf['Country'].unique()
countryReplaced = Maindf.groupby('Country').count()
countryReplaced = countryReplaced[countryReplaced['Age'] < 3].index
Maindf['Country'].replace(countryReplaced, 'other', inplace=True)

le_Country = pp.LabelEncoder()
Maindf['Country'] = le_Country.fit_transform(Maindf['Country']) 
Maindf['Country'].unique()

# pd.DataFrame(Maindf).head().to_csv('testasdf.csv')


# #Label encoding for feature Gender of test data
Testdf['Gender'] = le_Gender.transform(Testdf['Gender']) 

# #Label encoding for feature University Degree of test data
Testdf['University Degree'] = le_UniDeg.transform(Testdf['University Degree']) 

# # #Label encoding for feature Hair Color of test data
Testdf['Hair Color'] = le_HairClr.transform(Testdf['Hair Color']) 


#Removed the feature Income in EUR from data frame and equated to Y
Y1 = Testdf['Income']
#print(Y)
Testdf = Testdf.drop(columns=['Income'])
#print(Testdf.head())

# Label encoding for feature Profession of test data
testProfessionList = Testdf['Profession'].unique()
encodedProfession = list(set(professionList) - set(professionReplaced))
testProfessionReplace = list(set(testProfessionList) - set(encodedProfession))
Testdf['Profession'] = Testdf['Profession'].replace(testProfessionReplace, 'other profession')

Testdf['Profession'] = le_Prof.transform(Testdf['Profession']) 

#Label encoding for feature Country of test data
testCountryList = Testdf['Country'].unique()
encodedCountries = list(set(countryList) - set(countryReplaced))
testCountryReplace = list(set(testCountryList) - set(encodedCountries))
Testdf['Country'] = Testdf['Country'].replace(testCountryReplace, 'other')

Testdf['Country'] = le_Country.transform(Testdf['Country']) 

Maindf['Gender'],Testdf['Gender']=target_encode(Maindf['Gender'],Testdf['Gender'],Y)
Maindf['University Degree'],Testdf['University Degree']=target_encode(Maindf['University Degree'],Testdf['University Degree'],Y)
Maindf['Hair Color'],Testdf['Hair Color']=target_encode(Maindf['Hair Color'],Testdf['Hair Color'],Y)
Maindf['Profession'],Testdf['Profession']=target_encode(Maindf['Profession'],Testdf['Profession'],Y)
Maindf['Country'],Testdf['Country']=target_encode(Maindf['Country'],Testdf['Country'],Y)

# Training the DATA
X_train, X_test, y_train, y_test = train_test_split(Maindf, Y, test_size=0.2, random_state=0)

regressor = RandomForestRegressor(n_estimators = 100, random_state = 42)  
regressor.fit(X_train, y_train) #training the algorithm

y_pred = regressor.predict(X_test)

print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))  
print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))  
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))

Instance = Testdf['Instance']
Instance = pd.DataFrame(Instance, columns=['Instance'])


y_pred1 = regressor.predict(Testdf)

Income = pd.DataFrame(y_pred1,columns=['Income'])

file = Instance.join(Income)
file.to_csv('testoutput.csv',index=False)

