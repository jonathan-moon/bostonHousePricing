# Importing the libraries
import pandas as pd
import numpy as np
from sklearn import metrics
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline
from sklearn.metrics import confusion_matrix

# Linear Regression
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

#Adaboost
from sklearn.ensemble import AdaBoostRegressor
from collections import defaultdict
from sklearn.tree import DecisionTreeRegressor

#import the data
data = pd.read_csv('housing_correct.csv', header=None)

#add headers
data.columns = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT', 'PRICE']

#LINEAR REGRESSION
X = data.drop(columns='PRICE', axis=1)
y = data['PRICE']
lin_reg = LinearRegression()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.3, random_state=42)
lin_reg.fit(X_train, y_train)
lin_reg_predict = lin_reg.predict(X_test) #predictions
accuracy = lin_reg.score(X_test, y_test) #accuracies
print("Linear regression accuracy:", accuracy)

# Visualizing the differences between actual prices and predicted values
plt.scatter(y_test, lin_reg_predict)
plt.xlabel("Prices")
plt.ylabel("Predicted Prices")
plt.title("Prices vs Predicted Prices")
plt.savefig("lin_predictions.svg")
plt.show()

#Residual plot
plt.scatter(lin_reg_predict,y_test-lin_reg_predict)
plt.title("Predicted vs Residuals Linear Regression")
plt.xlabel("Predicted")
plt.ylabel("Residuals")
plt.savefig("lin_residuals.svg")
plt.show()

#FEATUER IMPORTANCE GRAPH
coefficient = lin_reg.coef_
fig, ax = plt.subplots()
sns.barplot(x=coefficient, y=X.columns, ax=ax)
plt.xlabel("Coefficient")
plt.ylabel("Feature")
plt.title("Linear Regression Feature Importance")
plt.subplots_adjust(left=0.4)
plt.savefig("linreg_features.svg")
plt.show()

# def plot_confusion_matrix(matrix):
#     fig = plt.figure(figsize=(8,8))
#     ax = fig.add_subplot(111)
#     cax = ax.matshow(matrix)
#     fig.colorbar(cax)

# confusion_matrix = metrics.confusion_matrix(y_test, lin_reg_predict)
# plot_confusion_matrix(confusion_matrix)
# plt.show()

#ADABOOST
depths = [1, 2, 4, 6, 8, 10, 12]
estimators = [10, 25, 50, 100, 200, 400, 800]
accuracies = []
accuracies_frame = pd.DataFrame()
scores = defaultdict(dict)
regressors = defaultdict(dict)
i = 0
for depth in depths:
    depth_accuracies=[]
    for max_estimators in estimators:
        regressor = AdaBoostRegressor(
            DecisionTreeRegressor(max_depth=depth),
            n_estimators=max_estimators,
            learning_rate=1.0)
        regressor.fit(X_train, y_train)
        regressors[depth][max_estimators] = regressor
        scores[depth][max_estimators] = regressor.score(X_test, y_test)
        accuracies.append(scores[depth][max_estimators])
        depth_accuracies.append(regressor.score(X_test, y_test))
        print("Depth: " + str(depth) + " Estimator: " + str(max_estimators))
        print("Accuracy = " + str(scores[depth][max_estimators]))
    accuracies_frame = accuracies_frame.append(pd.Series(depth_accuracies,
                                                            index=estimators,
                                                            name=str(depths[i]) + " depth"))
    accuracies_frame.transpose().plot(style='.-')
    plt.xlabel("Estimators")
    plt.ylabel("Accuracys")
    plt.savefig("800learners.svg")
    i+=1

#Find "best" adaboost regressor
best_score = 0
best_depth = 0
best_estimator = 0
for depth in scores:
    for estimator in scores[depth]:
        if scores[depth][estimator] > best_score:
            best_score = scores[depth][estimator]
            best_depth = depth
            best_estimator = estimator
best_ada = regressors[best_depth][best_estimator]
print("best depth is: ", best_depth)
print("best estimator is: ", best_estimator)
best_ada_pred = best_ada.predict(X_test)
ada_residuals = y_test - best_ada_pred


#Feature Importance for "Best" Model
importances = best_ada.feature_importances_
indices = np.argsort(importances)[::-1]
fig, ax = plt.subplots()
sns.barplot(x=importances[indices], y=X.columns[indices])
plt.xlabel("Feature Importance")
plt.ylabel("Features")
plt.title("Feature Importances of Best AdaBoost Model")
plt.subplots_adjust(left=0.2)
plt.savefig("feature_importance_ada.svg")
plt.show()

#Residual Plot for "Best" Model
plt.scatter(best_ada_pred, ada_residuals)
plt.xlabel('Predicted Values')
plt.ylabel('Residuals')
plt.title('Residual Plot for Adaboost Regression')
plt.savefig("residual_plot_ada.svg")
plt.show()

#Heat <ap for Adaboost

import seaborn as sns

# Convert scores dictionary to numpy array for heatmap visualization
heatmap_data = np.array([[scores[d][e] for e in estimators] for d in depths])

# Create heatmap with seaborn
sns.heatmap(heatmap_data, annot=True, fmt=".4f", xticklabels=estimators, yticklabels=depths, cmap="YlGnBu")

# Set plot labels and title
plt.xlabel("Number of Estimators")
plt.ylabel("Max Depth")
plt.title("AdaBoost Performance on Boston Housing")
plt.savefig("heat_map_800.svg")

