'''# CRISP-ML(Q)
Business Problem: 1.	Solar power consumption has been recorded by city councils at regular intervals. The reason behind doing so is to understand how businesses are using solar power so that they can cut down on nonrenewable sources of energy and shift towards renewable energy. Based on the data, build a forecasting model, and provide insights on it. 

Business Objective: Maximize Solar Power Consumption
Business Constraints: Minimize the production cost of renewable sources

Success Criteria: 
    Business: Increase the production of renewable energy at least 20%
    ML: Achieve an accuracy of at least 85%
    Economic: Achieve an increase in revenue by at least $200K

Data Understanding:
Feature	Description	Type	Relevance
date	Date of power consumption	Quantitative	Relevant
cum_power	Cumulative power consumption	Quantitative	Relevant
'''    

import pandas as pd
import numpy as np
# import pickle
from sqlalchemy import create_engine, text

user = 'root' # user name
pw = '1234' # password
db = 'solar_db' # database
# creating engine to connect database
engine = create_engine(f"mysql+pymysql://{user}:{pw}@localhost/{db}")

df = pd.read_csv(r"C:/Users/Lenovo/Downloads/Study material/Data Science/Forecasting/Assignment/solarpower_cumuldaybyday2.csv")

# dumping data into database 
df.to_sql('solar', con = engine, if_exists = 'replace', chunksize = 1000, index = False)


# loading data from database
sql = 'select * from solar'

solar = pd.read_sql_query(text(sql), con = engine.connect() )

print(solar)
solar.shape

# Data Pre-processing
solar["t"] = np.arange(1, 2559) # Linear Trend is captured
solar["t_square"] = solar["t"] * solar["t"] # Quadratic trend or polynomial with '2' degrees trend is captured
solar["log_cum_power"] = np.log(solar["cum_power"]) # Exponential trend is captured
solar.columns

solar.info()

p = solar["date"][0]
# Convert 'date' column to datetime format
solar['date'] = pd.to_datetime(solar['date'], format='%d/%m/%Y')

# Extract month from 'date' column
solar['month'] = solar['date'].dt.strftime('%b')


    
date_dummies = pd.DataFrame(pd.get_dummies(solar['month']))
solar1 = pd.concat([solar, date_dummies], axis = 1)
solar1 = solar1.drop(columns = "month")

# Visualization - Time plot
solar1.cum_power.plot()

# Data Partition
Train = solar1.head(2193)
Test = solar1.tail(365)

# to change the index value in pandas data frame 
# Test.set_index(np.arange(1, 13))

####################### Linear ##########################
import statsmodels.formula.api as smf 

linear_model = smf.ols('cum_power ~ t', data = Train).fit()
linear_model.summary()
pred_linear =  pd.Series(linear_model.predict(pd.DataFrame(Test['t'])))
rmse_linear = np.sqrt(np.mean((np.array(Test['cum_power']) - np.array(pred_linear))**2))
rmse_linear

##################### Exponential ##############################

Exp = smf.ols('log_cum_power ~ t', data = Train).fit()
pred_Exp = pd.Series(Exp.predict(pd.DataFrame(Test['t'])))
rmse_Exp = np.sqrt(np.mean((np.array(Test['cum_power']) - np.array(np.exp(pred_Exp)))**2))
rmse_Exp

#################### Quadratic ###############################

Quad = smf.ols('cum_power ~ t + t_square', data = Train).fit()
pred_Quad = pd.Series(Quad.predict(Test[["t", "t_square"]]))
rmse_Quad = np.sqrt(np.mean((np.array(Test['cum_power']) - np.array(pred_Quad))**2))
rmse_Quad

################### Additive Seasonality ########################

add_sea = smf.ols('cum_power ~ Jan + Feb + Mar + Apr + May + Jun + Jul + Aug + Sep + Oct + Nov', data = Train).fit()
add_sea.summary()
pred_add_sea = pd.Series(add_sea.predict(Test[['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov']]))
rmse_add_sea = np.sqrt(np.mean((np.array(Test['cum_power']) - np.array(pred_add_sea))**2))
rmse_add_sea

################## Multiplicative Seasonality ##################

Mul_sea = smf.ols('log_cum_power ~ Jan + Feb + Mar + Apr + May + Jun + Jul + Aug + Sep + Oct + Nov', data = Train).fit()
pred_Mult_sea = pd.Series(Mul_sea.predict(Test[['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov']]))
rmse_Mult_sea = np.sqrt(np.mean((np.array(Test['cum_power']) - np.array(np.exp(pred_Mult_sea)))**2))
rmse_Mult_sea

################## Additive Seasonality Quadratic Trend ############################

add_sea_Quad = smf.ols('cum_power ~ t + t_square + Jan + Feb + Mar + Apr + May + Jun + Jul + Aug + Sep + Oct + Nov', data = Train).fit()
pred_add_sea_quad = pd.Series(add_sea_Quad.predict(Test[['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 't', 't_square']]))
rmse_add_sea_quad = np.sqrt(np.mean((np.array(Test['cum_power']) - np.array(pred_add_sea_quad))**2))
rmse_add_sea_quad 

################## Multiplicative Seasonality Linear Trend  ###########

Mul_sea_linear = smf.ols('log_cum_power ~ t + Jan + Feb + Mar + Apr + May + Jun + Jul + Aug + Sep + Oct + Nov', data = Train).fit()
pred_Mult_sea_linear = pd.Series(Mul_sea_linear.predict(Test[['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 't']]))
rmse_Mult_sea_linear = np.sqrt(np.mean((np.array(Test['cum_power']) - np.array(np.exp(pred_Mult_sea_linear)))**2))
rmse_Mult_sea_linear

################## Testing #######################################

data = {"MODEL":pd.Series(["rmse_linear", "rmse_Exp", "rmse_Quad", "rmse_add_sea", "rmse_Mult_sea", "rmse_add_sea_quad", "rmse_Mult_sea_linear"]), "RMSE_Values":pd.Series([rmse_linear, rmse_Exp, rmse_Quad, rmse_add_sea, rmse_Mult_sea, rmse_add_sea_quad, rmse_Mult_sea_linear])}
table_rmse = pd.DataFrame(data)
table_rmse

# 'rmse_add_sea_quad' has the least RMSE value among the models prepared so far. Use these features and build forecasting model using entire data
model_full = smf.ols('cum_power ~ t + t_square + Jan + Feb + Mar + Apr + May + Jun + Jul + Aug + Sep + Oct + Nov', data = solar1).fit()

predict_data = pd.read_excel(r"C:/Users/Lenovo/Downloads/Study material/Data Science/Forecasting/Assignment/Predict_new.xlsx")

pred_new  = pd.Series(model_full.predict(predict_data))
pred_new

predict_data["forecasted_cum_power"] = pd.Series(pred_new)


# The models and results have save and load method, so you don't need to use the pickle module directly.
# to save model
model_full.save("Reg_model.pickle")

import os
os.getcwd()

# to load model
from statsmodels.regression.linear_model import OLSResults
model = OLSResults.load("Reg_model.pickle")



# RESIDUALS MIGHT HAVE ADDITIONAL INFORMATION!

# Autoregression Model (AR)
# Calculating Residuals from best model applied on full data
# AV - FV
full_res = solar1.cum_power - model.predict(solar1)

# ACF plot on residuals
import statsmodels.graphics.tsaplots as tsa_plots
tsa_plots.plot_acf(full_res, lags = 12)
# ACF is an (complete) auto-correlation function gives values 
# of auto-correlation of any time series with its lagged values.

# PACF is a partial auto-correlation function. 
# It finds correlations of Y with lags of the residuals of the time series 
tsa_plots.plot_pacf(full_res, lags = 12)

# Alternative approach for ACF plot is explained in next 2 lines
# from pandas.plotting import autocorrelation_plot
# autocorrelation_ppyplot.show()
                          
# AR Autoregressive model
from statsmodels.tsa.ar_model import AutoReg
model_ar = AutoReg(full_res, lags = [1])
model_fit = model_ar.fit()

print('Coefficients: %s' % model_fit.params)

pred_res = model_fit.predict(start = len(full_res), end = len(full_res) + len(predict_data) - 1, dynamic = False)
pred_res.reset_index(drop = True, inplace = True)

# The Final Predictions using ASQT and AR(1) Model
final_pred = pred_new + pred_res
final_pred
