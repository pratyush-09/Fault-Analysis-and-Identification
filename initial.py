# importing the libraries
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# importing the dataset
dataset=pd.read_csv('data.csv',parse_dates=True)
imp=dataset.iloc[:,0]
time=dataset.iloc[:,3]
solar=dataset.iloc[:,4]
temp=dataset.iloc[:,5]
wind=dataset.iloc[:,6]
demand=dataset.iloc[:,7]

# data visualization
plt.scatter(time,solar)
plt.title('Time vs Solar')
plt.show()
plt.scatter(time,temp)
plt.title('Time vs Temperature')
plt.show()
plt.scatter(time,wind)
plt.title('Time vs Wind')
plt.show()
plt.scatter(time,demand)
plt.title('Time vs Demand')
plt.show()

plt.bar(time,solar)
plt.title('Time vs Solar')
plt.show()
plt.bar(time,temp)
plt.title('Time vs Temperature')
plt.show()
plt.bar(time,wind)
plt.title('Time vs Wind')
plt.show()
plt.bar(time,demand)
plt.title('Time vs Demand')
plt.show()

# transforming the data into arrays
imp,time,solar,temp,wind,demand=np.array(imp),np.array(time),np.array(solar),np.array(temp),np.array(wind),np.array(demand)

# reshaping the arrays into 1-D
imp=imp.reshape(-1,1)
time=time.reshape(-1,1)
solar=solar.reshape(-1,1)
temp=temp.reshape(-1,1)
wind=wind.reshape(-1,1)
demand=demand.reshape(-1,1)

# feature scaling
from sklearn.preprocessing import StandardScaler
sc_imp=StandardScaler()
sc_solar=StandardScaler()
sc_temp=StandardScaler()
sc_wind=StandardScaler()
sc_demand=StandardScaler()
scaled_imp=sc_imp.fit_transform(imp)
scaled_solar=sc_solar.fit_transform(solar)
scaled_temp=sc_temp.fit_transform(temp)
scaled_wind=sc_wind.fit_transform(wind)
scaled_demand=sc_wind.fit_transform(demand)
        
# splitting the dataset into training and test set
from sklearn.model_selection import train_test_split
time_train1, time_test1, solar_train, solar_test= train_test_split(imp, solar, test_size=0.1, random_state=0)
time_train2, time_test2, temp_train, temp_test= train_test_split(imp, temp, test_size=0.1, random_state=0)
time_train3, time_test3, wind_train, wind_test= train_test_split(imp, wind, test_size=0.1, random_state=0)
time_train4, time_test4, demand_train, demand_test= train_test_split(imp, demand, test_size=0.1, random_state=0)

# LINEAR REGRESSION

# fitting simple linear regression into training set
from sklearn.linear_model import LinearRegression
simple_linear1=LinearRegression()
simple_linear2=LinearRegression()
simple_linear3=LinearRegression()
simple_linear4=LinearRegression()
simple_linear1.fit(time_train1,solar_train)
simple_linear2.fit(time_train2,temp_train)
simple_linear3.fit(time_train3,wind_train)
simple_linear4.fit(time_train4,demand_train)

# prediction using linear regression
solar_pred1=simple_linear1.predict(time_test1)
temp_pred1=simple_linear2.predict(time_test2)
wind_pred1=simple_linear3.predict(time_test3)
demand_pred1=simple_linear4.predict(time_test4)

# evaluating the performance of the polynomial regresion model (FOR TIME VS SOLAR)
print("Acc(lin. reg (time vs solar training)):",simple_linear1.score(time_train1, solar_train))
print("Acc(lin. reg (time vs solar testing)):",simple_linear1.score(time_test1, solar_test))
# evaluating the performance of the polynomial regresion model (FOR TIME VS TEMP)
print("Acc(lin. reg (time vs temp training)):",simple_linear2.score(time_train2, temp_train))
print("Acc(lin. reg (time vs temp testing)):",simple_linear2.score(time_test2, temp_test))
# evaluating the performance of the polynomial regresion model (FOR TIME VS WIND)
print("Acc(lin. reg (time vs wind training)):",simple_linear3.score(time_train3, wind_train))
print("Acc(lin. reg (time vs wind testing)):",simple_linear3.score(time_test3, wind_test))
# evaluating the performance of the polynomial regresion model (FOR TIME VS DEMAND)
print("Acc(lin. reg (time vs demand training)):",simple_linear4.score(time_train4, demand_train))
print("Acc(lin. reg (time vs demand testing)):",simple_linear4.score(time_test4, demand_test))

# visualizing the linear regression model predictions
plt.scatter(imp, solar, color='red')
plt.plot(imp, simple_linear1.predict(imp), color='blue')
plt.title('Linear Regression (Time vs Solar)')
plt.show()
plt.scatter(imp, temp, color='red')
plt.plot(imp, simple_linear2.predict(imp), color='blue')
plt.title('Linear Regression (Time vs Temperature)')
plt.show()
plt.scatter(imp, wind, color='red')
plt.plot(imp, simple_linear3.predict(imp), color='blue')
plt.title('Linear Regression (Time vs Wind)')
plt.show()
plt.scatter(imp, demand, color='red')
plt.plot(imp, simple_linear4.predict(imp), color='blue')
plt.title('Linear Regression (Time vs Demand)')
plt.show()

# POLYNOMIAL REGRESSION

from sklearn.preprocessing import PolynomialFeatures

# fitting polynomial regression into training set (FOR TIME VS SOLAR)
poly_reg_init1=PolynomialFeatures(degree=7)
time_poly1=poly_reg_init1.fit_transform(imp)
time_train_poly1, time_test_poly1, solar_train_poly, solar_test_poly=train_test_split(time_poly1, solar, test_size=0.1, random_state=0)
poly_reg1=LinearRegression()
poly_reg1.fit(time_train_poly1,solar_train_poly)

# fitting polynomial regression into training set (FOR TIME VS TEMP)
poly_reg_init2=PolynomialFeatures(degree=7)
time_poly2=poly_reg_init2.fit_transform(imp)
time_train_poly2, time_test_poly2, temp_train_poly, temp_test_poly=train_test_split(time_poly2, temp, test_size=0.1, random_state=0)
poly_reg2=LinearRegression()
poly_reg2.fit(time_train_poly2,temp_train_poly)

# fitting polynomial regression into training set (FOR TIME VS WIND)
poly_reg_init3=PolynomialFeatures(degree=5)
time_poly3=poly_reg_init3.fit_transform(imp)
time_train_poly3, time_test_poly3, wind_train_poly, wind_test_poly=train_test_split(time_poly3, wind, test_size=0.1, random_state=0)
poly_reg3=LinearRegression()
poly_reg3.fit(time_train_poly3,wind_train_poly)

# fitting polynomial regression into training set (FOR TIME VS DEMAND)
poly_reg_init4=PolynomialFeatures(degree=5)
time_poly4=poly_reg_init4.fit_transform(imp)
time_train_poly4, time_test_poly4, demand_train_poly, demand_test_poly=train_test_split(time_poly4, demand, test_size=0.1, random_state=0)
poly_reg4=LinearRegression()
poly_reg4.fit(time_train_poly4,demand_train_poly)

# prediction using polynomial regression
solar_predict_poly=poly_reg1.predict(time_test_poly1)
temp_predict_poly=poly_reg2.predict(time_test_poly2)
wind_predict_poly=poly_reg3.predict(time_test_poly3)
demand_predict_poly=poly_reg4.predict(time_test_poly4)

# evaluating the performance of the polynomial regresion model (FOR TIME VS SOLAR)
print("Acc(pol. reg (time vs solar training)):",poly_reg1.score(time_train_poly1, solar_train_poly))
print("Acc(pol. reg (time vs solar testing)):",poly_reg1.score(time_test_poly1, solar_test_poly))
# evaluating the performance of the polynomial regresion model (FOR TIME VS TEMP)
print("Acc(pol. reg (time vs temp training)):",poly_reg2.score(time_train_poly2, temp_train_poly))
print("Acc(pol. reg (time vs temp testing)):",poly_reg2.score(time_test_poly2, temp_test_poly))
# evaluating the performance of the polynomial regresion model (FOR TIME VS WIND)
print("Acc(pol. reg (time vs wind training)):",poly_reg3.score(time_train_poly3, wind_train_poly))
print("Acc(pol. reg (time vs wind testing)):",poly_reg3.score(time_test_poly3, wind_test_poly))
# evaluating the performance of the polynomial regresion model (FOR TIME VS DEMAND)
print("Acc(pol. reg (time vs demand training)):",poly_reg4.score(time_train_poly4, demand_train_poly))
print("Acc(pol. reg (time vs demand testing)):",poly_reg4.score(time_test_poly4, demand_test_poly))

# visualizing the polynomial regression model predictions
plt.scatter(imp,solar,color='red')
plt.plot(imp,poly_reg1.predict(poly_reg_init1.fit_transform(imp)),color='blue')
plt.title('Polynomial Regression (Time vs Solar) degree:7')
plt.show()
plt.scatter(imp,temp,color='red')
plt.plot(imp,poly_reg2.predict(poly_reg_init2.fit_transform(imp)),color='blue')
plt.title('Polynomial Regression (Time vs Temp) degree:7')
plt.show()
plt.scatter(imp,wind,color='red')
plt.plot(imp,poly_reg3.predict(poly_reg_init3.fit_transform(imp)),color='blue')
plt.title('Polynomial Regression (Time vs Wind) degree:5')
plt.show()
plt.scatter(imp,demand,color='red')
plt.plot(imp,poly_reg4.predict(poly_reg_init4.fit_transform(imp)),color='blue')
plt.title('Polynomial Regression (Time vs Demand) degree:5')
plt.show()

# SVR

from sklearn.svm import SVR

# fitting polynomial regression into training and test set (FOR TIME VS SOLAR)
time_train1_svr, time_test1_svr, solar_train_svr, solar_test_svr=train_test_split(scaled_imp, scaled_solar, test_size=0.1, random_state=0)
svr_reg1=SVR(kernel='rbf')
svr_reg1.fit(time_train1_svr,solar_train_svr)

# fitting polynomial regression into training and test set (FOR TIME VS TEMP)
svr_reg2=SVR(kernel='rbf')
svr_reg2.fit(time_train2,temp_train)

# fitting polynomial regression into training and test set (FOR TIME VS WIND)
time_train3_svr, time_test3_svr, wind_train_svr, wind_test_svr=train_test_split(scaled_imp, scaled_wind, test_size=0.1, random_state=0)
svr_reg3=SVR(kernel='rbf')
svr_reg3.fit(time_train3_svr,wind_train_svr)

# fitting polynomial regression into training and test set (FOR TIME VS DEMAND)
time_train4_svr, time_test4_svr, demand_train_svr, demand_test_svr=train_test_split(scaled_imp, scaled_demand, test_size=0.1, random_state=0)
svr_reg4=SVR(kernel='rbf')
svr_reg4.fit(time_train4_svr,demand_train_svr)

# prediction using SVR
solar_pred_svr=svr_reg1.predict(time_test1_svr)
temp_pred_svr=svr_reg2.predict(time_test2)
wind_pred_svr=svr_reg3.predict(time_test3_svr)
demand_pred_svr=svr_reg4.predict(time_test4_svr)

# evaluating the performance of the SVR model (FOR TIME VS SOLAR)
print("Acc(svr reg (time vs solar training)):",svr_reg1.score(time_train1_svr, solar_train_svr))
print("Acc(svr reg (time vs solar testing)):",svr_reg1.score(time_test1_svr, solar_test_svr))
# evaluating the performance of the SVR model (FOR TIME VS TEMP)
print("Acc(svr reg (time vs temp training)):",svr_reg2.score(time_train2, temp_train))
print("Acc(svr reg (time vs temp testing)):",svr_reg2.score(time_test2, temp_test))
# evaluating the performance of the SVR model (FOR TIME VS WIND)
print("Acc(svr reg (time vs wind training)):",svr_reg3.score(time_train3_svr, wind_train_svr))
print("Acc(svr reg (time vs wind testing)):",svr_reg3.score(time_test3_svr, wind_test_svr))
# evaluating the performance of the SVR model (FOR TIME VS DEMAND)
print("Acc(svr reg (time vs demand training)):",svr_reg4.score(time_train4_svr, demand_train_svr))
print("Acc(svr reg (time vs demand testing)):",svr_reg4.score(time_test4_svr, demand_test_svr))

# visualizing the svr model predictions
plt.scatter(scaled_imp,scaled_solar,color='red')
plt.plot(scaled_imp,svr_reg1.predict(scaled_imp),color='blue')
plt.title('SVR Regression (Time vs Solar)')
plt.show()
plt.scatter(imp,temp,color='red')
plt.plot(imp,svr_reg2.predict(imp),color='blue')
plt.title('SVR Regression (Time vs Temp)')
plt.show()
plt.scatter(scaled_imp,scaled_wind,color='red')
plt.plot(scaled_imp,svr_reg3.predict(scaled_imp),color='blue')
plt.title('SVR Regression (Time vs Wind)')
plt.show()
plt.scatter(scaled_imp,scaled_demand,color='red')
plt.plot(scaled_imp,svr_reg4.predict(scaled_imp),color='blue')
plt.title('SVR Regression (Time vs Demand)')
plt.show()

# DECISION TREE REGRESSION

# fitting decision tree regression into training and test set
from sklearn.tree import DecisionTreeRegressor
dectree_reg1=DecisionTreeRegressor()
dectree_reg2=DecisionTreeRegressor()
dectree_reg3=DecisionTreeRegressor()
dectree_reg4=DecisionTreeRegressor()
dectree_reg1.fit(time_train1,solar_train)
dectree_reg2.fit(time_train2,temp_train)
dectree_reg3.fit(time_train3,wind_train)
dectree_reg4.fit(time_train4,demand_train)

# prediction using Decision Tree Regression
solar_pred_dectree=dectree_reg1.predict(time_test1)
temp_pred_dectree=dectree_reg2.predict(time_test2)
wind_pred_dectree=dectree_reg3.predict(time_test3)
demand_pred_dectree=dectree_reg4.predict(time_test4)

# evaluating the performance of the Decision Tree regresion model (FOR TIME VS SOLAR)
print("Acc(dec. tree reg (time vs solar training)):",dectree_reg1.score(time_train1, solar_train))
print("Acc(dec. tree reg (time vs solar testing)):",dectree_reg1.score(time_test1, solar_test))
# evaluating the performance of the Decision Tree regresion model (FOR TIME VS TEMP)
print("Acc(dec. tree reg (time vs temp training)):",dectree_reg2.score(time_train2, temp_train))
print("Acc(dec. tree reg (time vs temp testing)):",dectree_reg2.score(time_test2, temp_test))
# evaluating the performance of the Decision Tree regresion model (FOR TIME VS WIND)
print("Acc(dec. tree reg (time vs wind training)):",dectree_reg3.score(time_train3, wind_train))
print("Acc(dec. tree reg (time vs wind testing)):",dectree_reg3.score(time_test3, wind_test))
# evaluating the performance of the Decision Tree regresion model (FOR TIME VS DEMAND)
print("Acc(dec. tree reg (time vs demand training)):",dectree_reg4.score(time_train4, demand_train))
print("Acc(dec. tree reg (time vs demand testing)):",dectree_reg4.score(time_test4, demand_test))

# visualizing the decision tree regression model predictions
plt.scatter(imp, solar, color='red')
plt.plot(imp, dectree_reg1.predict(imp),color='blue')
plt.title('Decision Tree Regression (Time vs Solar)')
plt.show()
plt.scatter(imp, temp, color='red')
plt.plot(imp, dectree_reg2.predict(imp),color='blue')
plt.title('Decision Tree Regression (Time vs Temp)')
plt.show()
plt.scatter(imp, wind, color='red')
plt.plot(imp, dectree_reg3.predict(imp),color='blue')
plt.title('Decision Tree Regression (Time vs Wind)')
plt.show()
plt.scatter(imp, demand, color='red')
plt.plot(imp, dectree_reg4.predict(imp),color='blue')
plt.title('Decision Tree Regression (Time vs Demand)')
plt.show()

# RANDOM FOREST REGRESSION

# fitting decision tree regression into training and test set
from sklearn.ensemble import RandomForestRegressor
ranforest_reg1=RandomForestRegressor(n_estimators=90,random_state=0)
ranforest_reg2=RandomForestRegressor(n_estimators=100,random_state=0)
ranforest_reg3=RandomForestRegressor(n_estimators=100,random_state=0)
ranforest_reg4=RandomForestRegressor(n_estimators=100,random_state=0)
ranforest_reg1.fit(time_train1, solar_train)
ranforest_reg2.fit(time_train2, temp_train)
ranforest_reg3.fit(time_train3, wind_train)
ranforest_reg4.fit(time_train4, demand_train)

solar_pred_ranforest=ranforest_reg1.predict(time_test1)
temp_pred_ranforest=ranforest_reg2.predict(time_test2)
wind_pred_ranforest=ranforest_reg3.predict(time_test3)
demand_pred_ranforest=ranforest_reg4.predict(time_test4)

# evaluating the performance of the Random Forest regresion model (FOR TIME VS SOLAR)
print("Acc(ran. forest reg (time vs solar training)):",ranforest_reg1.score(time_train1, solar_train))
print("Acc(ran. forest reg (time vs solar testing)):",ranforest_reg1.score(time_test1, solar_test))
# evaluating the performance of the Random Forest regresion model (FOR TIME VS TEMP)
print("Acc(ran. forest reg (time vs temp training)):",ranforest_reg2.score(time_train2, temp_train))
print("Acc(ran. forest reg (time vs temp testing)):",ranforest_reg2.score(time_test2, temp_test))
# evaluating the performance of the Random Forest regresion model (FOR TIME VS WIND)
print("Acc(ran. forest reg (time vs wind training)):",ranforest_reg3.score(time_train3, wind_train))
print("Acc(ran. forest reg (time vs wind testing)):",ranforest_reg3.score(time_test3, wind_test))
# evaluating the performance of the Random Forest regresion model (FOR TIME VS DEMAND)
print("Acc(ran. forest reg (time vs demand training)):",ranforest_reg4.score(time_train4, demand_train))
print("Acc(ran. forest reg (time vs demand testing)):",ranforest_reg4.score(time_test4, demand_test))

# visualizing the random forest regression predictions
plt.scatter(imp,solar,color='red')
plt.plot(imp,ranforest_reg1.predict(imp),color='blue')
plt.title('Random Forest Regression (Time vs Solar)')
plt.show()
plt.scatter(imp,temp,color='red')
plt.plot(imp,ranforest_reg2.predict(imp),color='blue')
plt.title('Random Forest Regression (Time vs Temp)')
plt.show()
plt.scatter(imp,wind,color='red')
plt.plot(imp,ranforest_reg3.predict(imp),color='blue')
plt.title('Random Forest Regression (Time vs Wind)')
plt.show()
plt.scatter(imp,demand,color='red')
plt.plot(imp,ranforest_reg4.predict(imp),color='blue')
plt.title('Random Forest Regression (Time vs Demand)')
plt.show()


# BEST REGRESSION MODELS
print("\n\n")
print("Best Regression Models:-")
print("--for TIME vs SOLAR--")
print("Acc(lin. reg (time vs solar training)):",simple_linear1.score(time_train1, solar_train))
print("Acc(lin. reg (time vs solar testing)):",simple_linear1.score(time_test1, solar_test))
print("Acc(pol. reg (time vs solar training)):",poly_reg1.score(time_train_poly1, solar_train_poly))
print("Acc(pol. reg (time vs solar testing)):",poly_reg1.score(time_test_poly1, solar_test_poly))
print("Acc(svr reg (time vs solar training)):",svr_reg1.score(time_train1_svr, solar_train_svr))
print("Acc(svr reg (time vs solar testing)):",svr_reg1.score(time_test1_svr, solar_test_svr))
print("Acc(dec. tree reg (time vs solar training)):",dectree_reg1.score(time_train1, solar_train))
print("Acc(dec. tree reg (time vs solar testing)):",dectree_reg1.score(time_test1, solar_test))
print("Acc(ran. forest reg (time vs solar training)):",ranforest_reg1.score(time_train1, solar_train))
print("Acc(ran. forest reg (time vs solar testing)):",ranforest_reg1.score(time_test1, solar_test))
print("\n\n")
print("--for TIME vs TEMP--")
print("Acc(lin. reg (time vs temp training)):",simple_linear2.score(time_train2, temp_train))
print("Acc(lin. reg (time vs temp testing)):",simple_linear2.score(time_test2, temp_test))
print("Acc(pol. reg (time vs temp training)):",poly_reg2.score(time_train_poly2, temp_train_poly))
print("Acc(pol. reg (time vs temp testing)):",poly_reg2.score(time_test_poly2, temp_test_poly))
print("Acc(svr reg (time vs temp training)):",svr_reg2.score(time_train2, temp_train))
print("Acc(svr reg (time vs temp testing)):",svr_reg2.score(time_test2, temp_test))
print("Acc(dec. tree reg (time vs temp training)):",dectree_reg2.score(time_train2, temp_train))
print("Acc(dec. tree reg (time vs temp testing)):",dectree_reg2.score(time_test2, temp_test))
print("Acc(ran. forest reg (time vs temp training)):",ranforest_reg2.score(time_train2, temp_train))
print("Acc(ran. forest reg (time vs temp testing)):",ranforest_reg2.score(time_test2, temp_test))
print("\n\n")
print("--for TIME vs WIND--")
print("Acc(lin. reg (time vs wind training)):",simple_linear3.score(time_train3, wind_train))
print("Acc(lin. reg (time vs wind testing)):",simple_linear3.score(time_test3, wind_test))
print("Acc(pol. reg (time vs wind training)):",poly_reg3.score(time_train_poly3, wind_train_poly))
print("Acc(pol. reg (time vs wind testing)):",poly_reg3.score(time_test_poly3, wind_test_poly))
print("Acc(svr reg (time vs wind training)):",svr_reg3.score(time_train3_svr, wind_train_svr))
print("Acc(svr reg (time vs wind testing)):",svr_reg3.score(time_test3_svr, wind_test_svr))
print("Acc(dec. tree reg (time vs wind training)):",dectree_reg3.score(time_train3, wind_train))
print("Acc(dec. tree reg (time vs wind testing)):",dectree_reg3.score(time_test3, wind_test))
print("Acc(ran. forest reg (time vs demand training)):",ranforest_reg4.score(time_train4, demand_train))
print("Acc(ran. forest reg (time vs demand testing)):",ranforest_reg4.score(time_test4, demand_test))
print("\n\n")
print("--for TIME vs DEMAND--")
print("Acc(lin. reg (time vs demand training)):",simple_linear4.score(time_train4, demand_train))
print("Acc(lin. reg (time vs demand testing)):",simple_linear4.score(time_test4, demand_test))
print("Acc(pol. reg (time vs demand training)):",poly_reg4.score(time_train_poly4, demand_train_poly))
print("Acc(pol. reg (time vs demand testing)):",poly_reg4.score(time_test_poly4, demand_test_poly))
print("Acc(svr reg (time vs demand training)):",svr_reg4.score(time_train4_svr, demand_train_svr))
print("Acc(svr reg (time vs demand testing)):",svr_reg4.score(time_test4_svr, demand_test_svr))
print("Acc(dec. tree reg (time vs demand training)):",dectree_reg4.score(time_train4, demand_train))
print("Acc(dec. tree reg (time vs demand testing)):",dectree_reg4.score(time_test4, demand_test))
print("Acc(ran. forest reg (time vs demand training)):",ranforest_reg4.score(time_train4, demand_train))
print("Acc(ran. forest reg (time vs demand testing)):",ranforest_reg4.score(time_test4, demand_test))
print("\n\n")
print("TIME vs SOLAR: Decision Tree Regression (98.899%)")
print("TIME vs TEMP: Random Forest Regression (99.988%)")
print("TIME vs WIND: Decision Tree Regression (69.428%)")
print("TIME vs DEMAND: Random Forest Regression (52.208%)")