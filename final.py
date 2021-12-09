# importing the libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor

# importing the dataset
dataset=pd.read_csv('data.csv',parse_dates=True)
imp=dataset.iloc[:,0]
time=dataset.iloc[:,3]
solar=dataset.iloc[:,4]
temp=dataset.iloc[:,5]
wind=dataset.iloc[:,6]
demand=dataset.iloc[:,7]

# transforming the data into arrays
imp,time,solar,temp,wind,demand=np.array(imp),np.array(time),np.array(solar),np.array(temp),np.array(wind),np.array(demand)

# reshaping the arrays into 1-D
imp=imp.reshape(-1,1)
time=time.reshape(-1,1)
solar=solar.reshape(-1,1)
temp=temp.reshape(-1,1)
wind=wind.reshape(-1,1)
demand=demand.reshape(-1,1)



# FINAL MODEL BUILDING

# TIME vs SOLAR
time_train_solar, time_test_solar, solar_train_final, solar_test_final=train_test_split(imp,solar,test_size=0.1,random_state=9)
solar_regressor=DecisionTreeRegressor()
solar_regressor.fit(time_train_solar,solar_train_final)
solar_pred=solar_regressor.predict(imp)
print("Solar Prediction Accuracy: ", solar_regressor.score(time_test_solar,solar_test_final))

# TIME vs TEMP
time_train_temp, time_test_temp, temp_train_final, temp_test_final=train_test_split(imp,temp,test_size=0.1,random_state=9)
temp_regressor=RandomForestRegressor(n_estimators=90,random_state=0)
temp_regressor.fit(time_train_temp,temp_train_final)
temp_pred=temp_regressor.predict(imp)
print("Temperature Prediction Accuracy: ", temp_regressor.score(time_test_temp,temp_test_final))

# TIME vs WIND
time_train_wind, time_test_wind, wind_train_final, wind_test_final=train_test_split(imp,wind,test_size=0.1,random_state=0)
wind_regressor=DecisionTreeRegressor()
wind_regressor.fit(time_train_wind,wind_train_final)
wind_pred=wind_regressor.predict(imp)
print("Wind Prediction Accuracy: ", wind_regressor.score(time_test_wind,wind_test_final))

# TIME vs DEMAND
time_train_demand, time_test_demand, demand_train_final, demand_test_final=train_test_split(imp,demand,test_size=0.1,random_state=1)
demand_regressor=RandomForestRegressor(n_estimators=100,random_state=0)
demand_regressor.fit(time_train_demand,demand_train_final)
demand_pred=demand_regressor.predict(imp)
print("Demand Prediction Accuracy: ", demand_regressor.score(time_test_demand,demand_test_final))


# FAULT IDENTIFICATION AND CORRECTION
def check_fault():
    dataset_modified=pd.read_csv('data_modified.csv',parse_dates=True)
    solar_modified=dataset_modified.iloc[:,4]
    temp_modified=dataset_modified.iloc[:,5]
    wind_modified=dataset_modified.iloc[:,6]
    demand_modified=dataset_modified.iloc[:,7]
    
    # fault in solar
    solar_counter=0
    print("\n\n\nFAULT SOLAR DATA INJECTED AT:")
    for i in range(len(solar)):
        diff=solar_modified[i]-solar_pred[i]
        if(abs(diff)>100):
            solar_counter+=1
            correct_data=solar_pred[i]
            if(i>=551):
                i+=1
            if(i==1440):
                i=0
            hr=(i)//60
            minute=i%60
            if(minute<10):
                minute='0'+str(minute)
            if(hr<12):
                zone='AM'
            else:
                zone='PM'
            if(hr==0):
                hr=12
            elif(hr!=0 and hr<10):
                hr='0'+str(hr)
            elif(hr>12 and hr<22):
                hr='0'+str(hr%12)
            elif(hr>=22):
                hr=str(hr%12)
            time=str(hr)+":"+str(minute)+" "+zone
            print(time,", correct data: ",round(correct_data,2))
    if(solar_counter==0):
        print("No fault data found!!!")
        
        
    # fault in temperature
    temp_counter=0
    print("\n\n\nFAULT TEMPERATURE DATA INJECTED AT:")
    for i in range(len(solar)):
        diff=temp_modified[i]-temp_pred[i]
        if(abs(diff)>2.5):
            temp_counter+=1
            correct_data=temp_pred[i]
            if(i>=551):
                i+=1
            if(i==1440):
                i=0
            hr=(i)//60
            minute=i%60
            if(minute<10):
                minute='0'+str(minute)
            if(hr<12):
                zone='AM'
            else:
                zone='PM'
            if(hr==0):
                hr=12
            elif(hr!=0 and hr<10):
                hr='0'+str(hr)
            elif(hr>12 and hr<22):
                hr='0'+str(hr%12)
            elif(hr>=22):
                hr=str(hr%12)
            time=str(hr)+":"+str(minute)+" "+zone
            print(time,", correct data: ",round(correct_data,2))
    if(temp_counter==0):
        print("No fault data found!!!")
        
    # fault in wind
    wind_counter=0
    print("\n\n\nFAULT WIND DATA INJECTED AT:")
    for i in range(len(solar)):
        diff=wind_modified[i]-wind_pred[i]
        if(abs(diff)>10):
            wind_counter+=1
            correct_data=wind_pred[i]
            if(i>=551):
                i+=1
            if(i==1440):
                i=0
            hr=(i)//60
            minute=i%60
            if(minute<10):
                minute='0'+str(minute)
            if(hr<12):
                zone='AM'
            else:
                zone='PM'
            if(hr==0):
                hr=12
            elif(hr!=0 and hr<10):
                hr='0'+str(hr)
            elif(hr>12 and hr<22):
                hr='0'+str(hr%12)
            elif(hr>=22):
                hr=str(hr%12)
            time=str(hr)+":"+str(minute)+" "+zone
            print(time,", correct data: ",round(correct_data,2))
    if(wind_counter==0):
        print("No fault data found!!!")
    
    # fault in demand
    demand_counter=0
    print("\n\n\nFAULT DEMAND DATA INJECTED AT:")
    for i in range(len(solar)):
        diff=demand_modified[i]-demand_pred[i]
        if(abs(diff)>5):
            demand_counter+=1
            correct_data=demand_pred[i]
            if(i>=551):
                i+=1
            if(i==1440):
                i=0
            hr=(i)//60
            minute=i%60
            if(minute<10):
                minute='0'+str(minute)
            if(hr<12):
                zone='AM'
            else:
                zone='PM'
            if(hr==0):
                hr=12
            elif(hr!=0 and hr<10):
                hr='0'+str(hr)
            elif(hr>12 and hr<22):
                hr='0'+str(hr%12)
            elif(hr>=22):
                hr=str(hr%12)
            time=str(hr)+":"+str(minute)+" "+zone
            print(time,", correct data: ",round(correct_data,2))
    if(demand_counter==0):
        print("No fault data found!!!")