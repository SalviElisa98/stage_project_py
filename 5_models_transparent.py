# ELISA SALVI STAGE   classification of the trasparent caps
import mahotas
import glob
import cv2 as cv
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

path_folder = '/Users/davidesalvi/Desktop/stage_project_py'       # path folder stage_project_py
df_trasparent_notonly = pd.read_csv(path_folder + "/df_trasparent_notonly.csv")  # ONLY TRASPARENT CAPOFF

from sklearn import svm, datasets
from sklearn.svm import SVC, LinearSVC
from sklearn.neighbors import KNeighborsClassifier
import sklearn.model_selection as model_selection
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn import tree

# CALCULATE THE IMPORTANCE OF ALL THE FEATURE USING A LINEAR REGRESSION
XX = df_trasparent_notonly.iloc[:, 2:]  # drop list feature and useless feature:
XX = XX.drop(['Target', 'Path', 'channels', 'height', 'width', 'HOG', 'SIFT', 'TEXTURE', 'y_circle', 'x_circle', 'New Target', 'New Target Name', 'CircleArea', 'r_circle'], axis=1)
yy = df_trasparent_notonly["New Target"]
X_train, X_test, y_train, y_test = model_selection.train_test_split(XX, yy, train_size=0.70, test_size=0.30, random_state=101)  # divsione randomica in train e test
ss = StandardScaler()
X_train_scaled = ss.fit_transform(X_train)
#X_test_scaled = ss.transform(X_test)
model = LogisticRegression()
model.fit(X_train_scaled, y_train)
Importance_linreg = model.coef_[0]
Importance_linreg = abs(Importance_linreg)
importances_linreg_tot = pd.DataFrame(data={'Attribute': X_train.columns, 'Importance_linreg': Importance_linreg})

# CALCULATE THE IMPORTANCE OF A FEATURE USING A TREE-BASED MODEL
from xgboost import XGBClassifier
model2 = XGBClassifier()
model2.fit(X_train_scaled, y_train)
importances_linreg_tot['Importance_tree'] = model2.feature_importances_
importances_linreg_tot = importances_linreg_tot.sort_values(by='Importance_linreg', ascending=False)
importances_linreg_tot.to_csv(path_folder + '/images_notonlygood/test_models_trasparent/importances_tot_trasparent.csv')



# partitioning the dataset to applay algorithm (no list feature)
useful = ['New Target', 'Path', 'r_mean_circle', 'g_mean_circle', 'b_std_circle', 'SIFT_std', 'HOG_min', 'r_var_circle'] # only numeric, not list
df = df_trasparent_notonly.loc[:, [col for col in useful]]  # df used for the classification with only usefull 6 featureuseful = {list: 7} ['Target', 'Path', 'r_mean_circle', 'g_mean_circle', 'b_mean_circle', 'SIFT_mean', 'HOG_min']

# CLASSIFICATION ALGORITHM (NO LIST FEATURE AND TRASPARENT FLIPOFF GRUPPED)
# separate features set X from the target column (class label) y, and divide the data set to 80% for training, and 20% for testing:
X = df.iloc[:, 2:]
y = df["New Target"]  # raggruppamento dei trasparenti
X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, train_size=0.70, test_size=0.30, random_state=2023)  # divsione randomica in train e test

# SVM with Polynomial kernel
print('- SVM Polynomial Kernel')
poly = svm.SVC(kernel='poly', degree=3, C=1).fit(X_train, y_train)
poly_pred = poly.predict(X_test)
poly_accuracy = accuracy_score(y_test, poly_pred)   # accuracy ad f1 score
poly_f1 = f1_score(y_test, poly_pred, average='weighted')
print('Accuracy: ', "%.2f" % (poly_accuracy*100))
print('F1 score: ', "%.2f" % (poly_f1*100))


# Linear SVC
print('- Linear SVC')
linear_svc = LinearSVC().fit(X_train, y_train)
linear_svc_pred = linear_svc.predict(X_test)
acc_linear_svc = round(accuracy_score(y_test, linear_svc_pred) * 100, 2)
linear_svc_f1 = f1_score(y_test, linear_svc_pred, average='weighted')
print('Accuracy: ', "%.2f" % (acc_linear_svc))
print('F1 score: ', "%.2f" % (linear_svc_f1*100))


# KNN  k-Nearest Neighbors
print('- KNN:  k-Nearest Neighbors')
knn = KNeighborsClassifier(n_neighbors=3).fit(X_train, y_train)
knn_pred = knn.predict(X_test)
acc_knn = round(accuracy_score(y_test, knn_pred) * 100, 2)
knn_f1 = f1_score(y_test, knn_pred, average='weighted')
print('Accuracy: ', "%.2f" % (acc_knn))
print('F1 score: ', "%.2f" % (knn_f1*100))
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
result = confusion_matrix(y_test, knn_pred)
print('Confusion Matrix:')
print(result)
result1 = classification_report(y_test, knn_pred)
print('Classification Report:')
print(result1)
import seaborn as sns
ax = sns.heatmap(result, annot=True, cmap='Blues')  # plot confusion matrix
ax.set_title('KNN Confusion Matrix')
ax.set_xlabel('Predicted Values')
ax.set_ylabel('Actual Values')
names = ['Black-Gold', 'Red-Gold', 'Red-Silver', 'Black-Silver']
ax.xaxis.set_ticklabels(names)
ax.yaxis.set_ticklabels(names)
plt.savefig(path_folder + '/images_notonlygood/test_models_trasparent/knn_confusion_matrix_tras.png', dpi=300)
#EDA pairplot
g = sns.pairplot(df, hue='New Target', palette='Set2', markers=["o", "s", "D", 'd'], corner=True)
g.fig.legend(handles=g._legend_data.values(), labels=names, loc='upper center', ncol=4)
g.fig.savefig(path_folder + '/images_notonlygood/test_models_trasparent/eda_tras.png', dpi=300)


# Random Forest Classifier
print('- Random Forest Classifier')
random_forest = RandomForestClassifier(n_estimators=5).fit(X_train, y_train)
rf_pred = random_forest.predict(X_test)
acc_random_forest = round(accuracy_score(y_test, rf_pred) * 100, 2)
rf_f1 = f1_score(y_test, rf_pred, average='weighted')
print('Accuracy: ', "%.2f" % (acc_random_forest))
print('F1 score: ', "%.2f" % (rf_f1*100))
target_names = ['Black-Gold-250ml-1000ml', 'Red-Gold', 'Red-Silver', 'Black-Silver']
feature_names = X.columns
fig1, axes = plt.subplots(nrows=1, ncols=5, figsize=(180, 90))
for index in range(0, 5):
    tree.plot_tree(random_forest.estimators_[index], feature_names=feature_names, class_names=target_names, filled=True, ax=axes[index])
    axes[index].set_title('Estimator: ' + str(index), fontsize=150)
fig1.savefig(path_folder +'/images_notonlygood/test_models_trasparent/rf_5trees_tras.png')

# Decision Tree
print('- DECISION TREE')
decision_tree = DecisionTreeClassifier().fit(X_train, y_train)
DT_pred = decision_tree.predict(X_test)
acc_decision_tree = round(accuracy_score(y_test, DT_pred) * 100, 2)
DT_f1 = f1_score(y_test, DT_pred, average='weighted')
print('Accuracy: ', "%.2f" % (acc_decision_tree))
print('F1 score: ', "%.2f" % (DT_f1*100))
fig = plt.figure(figsize=(165, 85))
tree.plot_tree(decision_tree, feature_names=feature_names, class_names=target_names, filled=True)
fig.savefig(path_folder + '/images_notonlygood/test_models_trasparent/tree_trasparent.png')

# df of accuracy
models = pd.DataFrame({
    'Model': ['SVM Polynomial Kernel', 'Linear SVC', 'KNN', 'Random Forest', 'Decision Tree'],
    'Accuracy': [(poly_accuracy*100), acc_linear_svc, acc_knn, acc_random_forest, acc_decision_tree],
    'F1 score': [(poly_f1*100), (linear_svc_f1*100), (knn_f1*100), (rf_f1*100), (DT_f1*100)]
})
model = models.sort_values(by='Accuracy', ascending=False)
model.to_csv(path_folder + '/images_notonlygood/test_models_trasparent/model_accuracy_trasparent.csv')



path_list = []
path_all = df_trasparent_notonly['Path']
for ind in y_test.index:   # add image path in the y_test dataframe
    path_list.append(path_all[ind])

# df of ground truth and predictions
predizioni_test = pd.DataFrame({'Path_test': path_list, 'Target of y_test': y_test, "SVM Polynomial": poly_pred, 'Linear SVC':linear_svc_pred, 'KNN':knn_pred, 'Random Forest':rf_pred, 'Decision Tree':DT_pred})
predizioni_test.to_csv(path_folder + '/images_notonlygood/test_models_trasparent/predizioni_test_trasparent.csv')


names = ['Black-Gold-250ml-1000ml', 'Red-Gold', 'Red-Silver', 'Black-Silver']
# save images in folder test_models with the ground truth and the prediction of each model
predizioni = [poly_pred, linear_svc_pred, knn_pred, rf_pred, DT_pred]
predizioni_names = ['poly_pred', 'linear_svc_pred', 'knn_pred', 'rf_pred', 'DT_pred']
for model_pred, name_pred in zip(predizioni, predizioni_names):
    for i, gt in zip(range(len(path_list)), y_test):
        image = cv.imread(path_list[i], cv.IMREAD_COLOR)
        cv.putText(image, 'Prediction: ', (10, 60), cv.FONT_HERSHEY_SIMPLEX, 2.2, (0, 255, 255), 3)
        cv.putText(image, str(names[model_pred[i]]), (10, 150), cv.FONT_HERSHEY_SIMPLEX, 2.2, (0, 255, 255), 3)
        cv.putText(image, 'Ground Truth: ', (10, 250), cv.FONT_HERSHEY_SIMPLEX, 2.2, (0, 255, 0), 3)
        cv.putText(image, str(names[gt]), (10, 340), cv.FONT_HERSHEY_SIMPLEX, 2.2, (0, 255, 0), 3)
        name_list = path_list[i].split('/')
        image_path = path_folder + '/images_notonlygood/test_models_trasparent/' + str(name_pred) + '/' + name_list[-1]
        cv.imwrite(image_path, image)




# calculate the importance of the feature used
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

ss = StandardScaler()
X_train_scaled = ss.fit_transform(X_train)
X_test_scaled = ss.transform(X_test)

model = LogisticRegression()
model.fit(X_train_scaled, y_train)
importances_linreg = pd.DataFrame(data={'Attribute': X_train.columns, 'Importance': model.coef_[0]})
importances_linreg = importances_linreg.sort_values(by='Importance', ascending=False)
importances_linreg.to_csv(path_folder + '/images_notonlygood/test_models_trasparent/importances_linreg_trasparent.csv')



