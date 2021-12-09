# Fault-Analysis-and-Identification

The project is based on fault injection and identification in a data with solar, temperature, wind and power demand readings taken on each minute on a particular day.

In initial.py file,

Different plots were visualized to understand the relation between time and solar readings, time and temperature readings, time and wind readings and time and demand readings (present in data.csv file). In particular, scatter and bar plots are used to visualize the different relations.

Various machine learning models such as Linear Regression, Polynomial Regression, Support Vector Regression, Decision Tree Regression and Random Forest Regression are applied on each time vs solar, time vs temperature, time vs wind and time vs demand relations. From the 5 different models, the model with the maximum accuracy was selected for each case.


In final.py file,

The best model selected is used to train the ML model on each case and the predictions for the entire data is generated which gives us maximum accuracy.
Then the fault is injected in the data_modified file and the function is called and it displays the time at which the fault is injected and the corresponding predicted value for that time.
