# -*- coding: utf-8 -*-
"""
Created on Sat Jun 12 19:13:44 2021

@author: ASUS
"""

# Importing all the necessary libraries
import pandas as pd
import numpy as np
import seaborn as sns 
import statsmodels.api as sm
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf
from itertools import cycle
from statsmodels.tsa.arima_model import ARIMA
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.holtwinters import  Holt
from pandas.plotting import autocorrelation_plot
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.holtwinters import  SimpleExpSmoothing
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

# Loading the data file
file=pd.read_excel("D:\Data Science Assignments\Python-Assignment\Forecasting\CocaCola_Sales_Rawdata.xlsx")

# Modulating the dataset
file.Sales.plot()
seq = cycle(["Q1","Q2","Q3","Q4"])
file['Quarters'] = [next(seq) for count in range(file.shape[0])]
file=pd.get_dummies(file,columns=["Quarters"],prefix=["Q"])
file['t']=np.arange(1,43)
file['t_squared']=file['t']*file['t']
file['log_Sales']=np.log(file["Sales"])
file.head()
file.tail() 

# Dividing the dataframe into training and testing dataset
train=file.iloc[0:36,:]
test=file.iloc[36:40,]

# Root Mean Squared Error Function
def rmse(pred):
    return np.sqrt(np.mean((np.array(test["Sales"])-np.array(pred))**2))


######################## Model Based Approaches ###########################
    
# Linear Model
linear_mod=smf.ols('Sales~t',data=train).fit()
pred_lin=pd.Series(linear_mod.predict(test["t"]))
rmse_lin=rmse(pred_lin)  #Calculating RMSE Value

# Exponential Model
exponent_mod=smf.ols('log_Sales~t',data=train).fit()
pred_exp=pd.Series(np.exp(exponent_mod.predict(test)))
rmse_exp=rmse(pred_exp)  #Calculating RMSE Value

# Quadratic Model
quadratic_mod=smf.ols('Sales~t+t_squared',data=train).fit()
pred_qua=pd.Series(quadratic_mod.predict(test))
rmse_qua=rmse(pred_qua)  #Calculating RMSE Value

# Additive Seasonality Model
add_mod=smf.ols('Sales~Q_Q1+Q_Q2+Q_Q3',data=train).fit()
pred_add=pd.Series(add_mod.predict(test))
rmse_add=rmse(pred_add)  #Calculating RMSE Value

# Additive Seasonality With Quadratic Trend Model
addseaqua_mod=smf.ols('Sales~t+t_squared+Q_Q1+Q_Q2+Q_Q3',data=train).fit()
pred_adsequ=pd.Series(addseaqua_mod.predict(test))
rmse_adsqu=rmse(pred_adsequ)  #Calculating RMSE Value

# Multiplicative Seasonality Model
mul_mod=smf.ols('log_Sales~Q_Q1+Q_Q2+Q_Q3',data=train).fit()
pred_mul=pd.Series(np.exp(mul_mod.predict(test)))
rmse_mul=rmse(pred_mul)  #Calculating RMSE Value

# Multiplicative Seasonality With Additive Trend Model
mulad_mod=smf.ols('log_Sales~t+Q_Q1+Q_Q2+Q_Q3+Q_Q4',data=train).fit()
pred_mulad=pd.Series(np.exp(mulad_mod.predict(test)))
rmse_mulad=rmse(pred_mulad)  #Calculating RMSE Value

# Creating a table of models with their RMSE Values
RMSE_table=pd.DataFrame({'Models':["Linear Model","Exponential Model","Quadratic Model",
                         "Additive Seasonality Model",
                         "Additive Seasonality With Quadratic Trend Model",
                         "Multiplicative Seasonality Model",
                         "Multiplicative Seasonality With Additive Trend Model"],
                        'RMSE_Values':[rmse_lin,rmse_exp,rmse_qua,rmse_add,rmse_adsqu,rmse_mul
                         ,rmse_mulad]})
RMSE_table

# RMSE Value is lowest for Additive Seasonality With Quadratic Trend Model
adsq_mod=smf.ols('Sales~t+t_squared+Q_Q1+Q_Q2+Q_Q3',data=file).fit()
predicted_values=pd.Series(adsq_mod.predict(test))
rmse_value=rmse(predicted_values)
rmse_value

##########################################################################
# ARIMA and SARIMAX

file.head()
seq=cycle(["Q1","Q2","Q3","Q4"])
file["Quarters"]=[next(seq) for count in range(file.shape[0])]


# Augmented Dickey-Fuller Test
def adfuller_test(value):
    dftest=adfuller(value,autolag='AIC')
    dfoutput=pd.Series(dftest[0:4],index=["Test Statistic","P-Value",
                                          "Lags Used","Number of Observations Used"])
    return dfoutput

# Checking whether the dataset is stationary or not
adful=adfuller_test(file["Sales"])
adful # P-Value = 0.99

# First Differencing
file["Seasonal First Difference"]=file["Sales"]-(file["Sales"].shift(4))
# Checking the P-Value of the first difference
adful_difference1=adfuller_test(file["Seasonal First Difference"].dropna())
adful_difference1

# Second Differencing
file["Seasonal Second Difference"]=file["Sales"]-(file["Sales"].shift(8))
# Checking the P-Value of the Second difference
adful_difference2=adfuller_test(file["Seasonal Second Difference"].dropna())
adful_difference2

# Third Differencing
file["Seasonal Third Difference"]=file["Sales"]-(file["Sales"].shift(12))
# Checking the P-Value of the third difference
adful_difference3=adfuller_test(file["Seasonal Third Difference"].dropna())
adful_difference3

# Fourth Differencing
file["Seasonal Fourth Difference"]=file["Sales"]-(file["Sales"].shift(16))
# Checking the P-Value of the fourth difference
adful_difference4=adfuller_test(file["Seasonal Fourth Difference"].dropna())
adful_difference4 # P-Value=0.0023 which is lesser than 0.05

# Visualising the de-seasoned data
file["Seasonal Fourth Difference"].plot()

# Autocorrelation plot
autocorrelation_plot(file["Sales"])
plt.show()

# ACF Plot and PACF plot
fig=plt.figure(figsize=(12,8))
ax1=fig.add_subplot(211)
fig=plot_acf(file["Seasonal Fourth Difference"].dropna(),lags=20,ax=ax1)
ax2=fig.add_subplot(212)
fig=plot_pacf(file["Seasonal Fourth Difference"].dropna(), ax=ax2, lags=20)

# Applying SARIMAX function as the dataset is seasonal
model_sar=sm.tsa.statespace.SARIMAX(file["Sales"],order=(3,1,3),seasonal_order=(4,1,3,4))
result=model_sar.fit()
file["Forecast"]=result.predict(start=36,end=42,dynamic=True)
file[["Sales","Forecast"]].plot(figsize=(10,6))

# Forecasting the Future Sales
future=[file.index[-1]+count for count in range(1,11)]
future_df=pd.DataFrame(index=future[1:],columns=file.columns)
future_dataset=pd.concat([file,future_df])
future_dataset["Forecast"]=result.predict(start=41,end=51,dynamic=True)
future_dataset[["Sales","Forecast"]].plot(figsize=(12,8))

###########################################################################
# Data Driven Approaches
file.columns
file.index

heatmap=pd.pivot_table(data=file, aggfunc="mean", index=file.index, values="Sales",columns="Quarters",fill_value=0)
sns.heatmap(heatmap,annot=True,fmt="g")

#Visualisation of data as per every quarter
sns.boxplot(x="Quarters",y="Sales",data=file)
sns.lineplot(x="Quarters",y="Sales",data=file)

#Visualisation of the rolling average
file.Sales.plot(label="org")
for i in range(2,42,4):
    file["Sales"].rolling(i).mean().plot(label=str(i))
plt.legend(loc=3)

#Time Series Seasonal Decomposition Plot
decompose_ts_add=seasonal_decompose(file.Sales,model="additive",freq=4)
decompose_ts_add.plot()
decompose_ts_mul=seasonal_decompose(file.Sales,model="multiplicative",freq=4)
decompose_ts_mul.plot()

#Plotting ACF and PACF
plot_acf(file.Sales,lags=20)
plot_pacf(file.Sales,lags=20)

#MAPE Function
def MAPE(prediction):
    temp=(np.abs((prediction-test.Sales)/test.Sales)*100)
    return np.mean(temp)

# Simple Exponential Smoothing Model
ses_model=SimpleExpSmoothing(train["Sales"]).fit()
pred_ses=ses_model.predict(start=test.index[0], end=test.index[-1])
mape_ses=MAPE(pred_ses)

# Holt Model
hol_model=Holt(train["Sales"]).fit()
pred_hol=hol_model.predict(start=test.index[0],end=test.index[-1])
mape_hol=MAPE(pred_hol)

# Holt-Winter Exponential Smoothing Model with additive seasonality and additive trend
exp_add_ses=ExponentialSmoothing(train["Sales"],seasonal="add",trend="add",seasonal_periods=4).fit()
pred_add_ses=exp_add_ses.predict(start=test.index[0], end=test.index[-1])
mape_exp_add_ses=MAPE(pred_add_ses)

# Holt-Winter Exponential Smoothing Model with multiplicative seasonality and additive trend
exp_mul_ses=ExponentialSmoothing(train["Sales"],seasonal="mul",trend="add",seasonal_periods=4).fit()
pred_mul_ses=exp_mul_ses.predict(start=test.index[0],end=test.index[-1])
mape_exp_mul_ses=MAPE(pred_mul_ses)

# Creating a table of MAPE values
mape_table= pd.DataFrame({'Models':["Simple Esponential Smoothing Method","Holt Method",
                         "Exponential Smoothing Model with additive seasonality",
                         "Exponential Smoothing Model with Multiplicative Seasonality"],
                         'MAPE_Values':[mape_ses,mape_hol,mape_exp_add_ses,mape_exp_mul_ses]})
mape_table

# As MAPE score is less in Exponential Smoothing with multiplicative seasonality and additive trend
exp_model=ExponentialSmoothing(file["Sales"],seasonal="mul",trend="add",seasonal_periods=4).fit()
predict_exp=exp_model.predict(start=test.index[0],end=test.index[-1])
mape_score=MAPE(predict_exp)

#Visualisation of Predicted Values
sns.lineplot(x="Quarter",y="Sales",data=test)
sns.lineplot(x=test["Quarter"],y=predict_exp)
plt.legend(loc=2)

