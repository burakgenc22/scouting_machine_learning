# Kütüphane importları ve pd-set_option ayarları

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import GridSearchCV, cross_validate
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, VotingClassifier, GradientBoostingClassifier
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
from sklearn.svm import SVC
import warnings
warnings.simplefilter(action="ignore")

pd.set_option('display.max_columns', None)
pd.set_option('display.width', 170)
pd.set_option('display.max_rows', None)
pd.set_option('display.float_format', lambda x: '%.3f' % x)


# Veri setinin okutulması ve hazırlanması

df1 = pd.read_csv("scoutium_attributes.csv", delimiter=";")
df2 = pd.read_csv("scoutium_potential_labels.csv", delimiter=";")

dff = pd.merge(df1, df2, on=["task_response_id", "match_id", "evaluator_id", "player_id"], how="left")

## pozisyonu kaleci olan oyuncuların veri setinden çıkartılması
dff = dff.loc[~(dff["position_id"] == 1)]

## potential_label sütünunda below_avarage sınıfının frekansı çok düşük olduğu için veri setinden çıkartılması
dff["potential_label"].value_counts()
dff = dff.loc[~(dff["potential_label"] == "below_average")]

## pivot_table oluşturarak her satırda bir oyuncu olacak şekilde veri setinin düzenlenmesi
df = pd.pivot_table(dff, values="attribute_value", index=["player_id", "position_id", "potential_label"], columns=["attribute_id"])

df = df.reset_index()
df.head()

## sütunların tiplerinin string ifadelere çevirilmesi
df.columns = df.columns.astype(str)


#Encoding işlemleri

labelencoder = LabelEncoder()
df["potential_label"] = labelencoder.fit_transform(df["potential_label"])
df.head()
df["potential_label"].value_counts()

#Sayısal değişkenkenlerin yakalanması


num_cols = [col for col in df.columns if df[col].nunique() > 7]
num_cols = [col for col in num_cols if col not in "player_id"]
num_cols

#Standartlaştırma

standardscale = StandardScaler()
df[num_cols] = standardscale.fit_transform(df[num_cols])

df.head()

#Modelleme

y = df["potential_label"]
X = df.drop(["potential_label"], axis=1)

## Random forests

rf_model = RandomForestClassifier(random_state=17)

cv_results_1 = cross_validate(rf_model, X, y, cv=5, scoring=["accuracy", "f1", "roc_auc"])
cv_results_1["test_accuracy"].mean()
cv_results_1["test_f1"].mean()
cv_results_1["test_roc_auc"].mean()

# accuracy= 0.8635, f1= 0.5924, roc_auc= 0.9032

#Hiperparametre optimizasyonu
rf_model.get_params()

rf_params = {"max_depth": [5, 8, None],
             "max_features": [3, 5, 7, "auto"],
             "min_samples_split": [2, 5, 8, 15, 20],
             "n_estimators": [100, 200, 500]}

rf_best_grid = GridSearchCV(rf_model, rf_params, cv=5, n_jobs=-1, verbose=True).fit(X, y)

rf_final = rf_model.set_params(**rf_best_grid.best_params_, random_state=17).fit(X, y)

cv_results_1_1 = cross_validate(rf_final, X, y, cv=5, scoring=["accuracy", "f1", "roc_auc"])
cv_results_1_1["test_accuracy"].mean()
cv_results_1_1["test_f1"].mean()
cv_results_1_1["test_roc_auc"].mean()

# rf_final = accuracy= 0.8818, f1= 0.6148, roc_auc= 0.9096 / rf_model = accuracy= 0.8635, f1= 0.5924, roc_auc= 0.9032


#GBM

gbm_model = GradientBoostingClassifier(random_state=17)

cv_results_2 = cross_validate(gbm_model, X, y, cv=5, scoring=["accuracy", "f1", "roc_auc"])
cv_results_2["test_accuracy"].mean()
cv_results_2["test_f1"].mean()
cv_results_2["test_roc_auc"].mean()

# accuracy= 0.7939, f1= 0.5477, roc_auc= 0.8555


#Hiperparametre optimizasyonu
gbm_model.get_params()

gbm_params = {"learning_rate": [0.01, 0.1],
              "max_depth": [3, 8, 10],
              "n_estimators": [100, 500, 1000],
              "subsample": [1, 0.5, 0.7]}

gbm_best_grid = GridSearchCV(gbm_model, gbm_params, cv=5, n_jobs=-1, verbose=True).fit(X, y)

gbm_final = gbm_model.set_params(**gbm_best_grid.best_params_, random_state=17).fit(X, y)

cv_results_2_1 = cross_validate(gbm_final, X, y, cv=5, scoring=["accuracy", "f1", "roc_auc"])
cv_results_2_1["test_accuracy"].mean()
cv_results_2_1["test_f1"].mean()
cv_results_2_1["test_roc_auc"].mean()

# gbm_final = accuracy= 0.8858, f1= 0.6977, roc_auc= 0.8830 / gbm_model = accuracy= 0.7939, f1= 0.5477, roc_auc= 0.8555


#XGBOOST

xgboost_model = XGBClassifier(random_state=17)

cv_results_3 = cross_validate(xgboost_model, X, y, cv=5, scoring=["accuracy", "f1", "roc_auc"])
cv_results_3["test_accuracy"].mean()
cv_results_3["test_f1"].mean()
cv_results_3["test_roc_auc"].mean()

# accuracy= 0.8524, f1= 0.5853, roc_auc= 0.8434

#Hiperparametre optimizasyonu
xgboost_model.get_params()

xgboost_params = {"learning_rate": [0.1, 0.01],
                  "max_depth": [5, 8],
                  "n_estimators": [100, 500, 1000],
                  "colsample_bytree": [0.7, 1]}

xgboost_best_grid = GridSearchCV(xgboost_model, xgboost_params, cv=5, n_jobs=-1, verbose=True).fit(X, y)

xgboost_final = xgboost_model.set_params(**xgboost_best_grid.best_params_, random_state=17).fit(X, y)

cv_results_3_1 = cross_validate(xgboost_final, X, y, cv=5, scoring=["accuracy", "f1", "roc_auc"])
cv_results_3_1["test_accuracy"].mean()
cv_results_3_1["test_f1"].mean()
cv_results_3_1["test_roc_auc"].mean()

# xgboost_final = accuracy= 0.8819, f1= 0.6469, roc_auc= 0.8904 / xgboost_model = accuracy= 0.8524, f1= 0.5853, roc_auc= 0.8434


#LightGBM

lgbm_model = LGBMClassifier(random_state=17)

cv_results_4 = cross_validate(lgbm_model, X, y, cv=5, scoring=["accuracy", "f1", "roc_auc"])
cv_results_4["test_accuracy"].mean()
cv_results_4["test_f1"].mean()
cv_results_4["test_roc_auc"].mean()

# accuracy= 0.8708, f1= 0.6050, roc_auc= 0.8688

#Hiperparametre optimizasyonu

lgbm_model.get_params()

lgbm_params = {"learning_rate": [0.01, 0.1],
               "n_estimators": [100, 300, 500, 1000],
               "colsample_bytree": [0.5, 0.7, 1]}

lgbm_best_grid = GridSearchCV(lgbm_model, lgbm_params, cv=5, n_jobs=-1, verbose=True).fit(X, y)

lgbm_final = lgbm_model.set_params(**lgbm_best_grid.best_params_, random_state=17).fit(X, y)

cv_results_4_1 = cross_validate(lgbm_final, X, y, cv=5, scoring=["accuracy", "f1", "roc_auc"])
cv_results_4_1["test_accuracy"].mean()
cv_results_4_1["test_f1"].mean()
cv_results_4_1["test_roc_auc"].mean()

# lgbm_final = accuracy= 0.8856, f1= 0.6501, roc_auc= 0.8904 / lgbm_model = accuracy= 0.8708, f1= 0.6050, roc_auc= 0.8688


#CatBOOST

catboost_model = CatBoostClassifier(random_state=17)

cv_results_5 = cross_validate(catboost_model, X, y, cv=5, scoring=["accuracy", "f1", "roc_auc"])
cv_results_5["test_accuracy"].mean()
cv_results_5["test_f1"].mean()
cv_results_5["test_roc_auc"].mean()

# accuracy= 0.8745, f1= 0.5958, roc_auc= 0.8940


#Hiperparametre optimizasyonu
catboost_model.get_params()

catboost_params = {"iterations": [200, 500],
                   "learning_rate": [0.01, 0.1],
                   "depth": [3, 6]}

catboost_best_grid = GridSearchCV(catboost_model, catboost_params, cv=5, n_jobs=-1, verbose=False).fit(X, y)

catboost_final = catboost_model.set_params(**catboost_best_grid.best_params_, random_state=17).fit(X, y)

cv_results_5_1 = cross_validate(catboost_final, X, y, cv=5, scoring=["accuracy", "f1", "roc_auc"])
cv_results_5_1["test_accuracy"].mean()
cv_results_5_1["test_f1"].mean()
cv_results_5_1["test_roc_auc"].mean()

# catboost_final =  accuracy= 0.8781, f1= 0.5842, roc_auc= 0.8866 / catboost_model = accuracy= 0.8745, f1= 0.5958, roc_auc= 0.8940



#CART

cart_model = DecisionTreeClassifier(random_state=17)

cv_results_6 = cross_validate(cart_model, X, y, cv=5, scoring=["accuracy", "f1", "roc_auc"])
cv_results_6["test_accuracy"].mean()
cv_results_6["test_f1"].mean()
cv_results_6["test_roc_auc"].mean()

# accuracy= 0.7638, f1= 0.5428, roc_auc= 0.7309


#Hiperparametre optimizasyonu

cart_model.get_params()

cart_params = {'max_depth': range(1, 11),
               "min_samples_split": range(2, 20)}

cart_best_grid = GridSearchCV(cart_model,
                              cart_params,
                              cv=5,
                              n_jobs=-1,
                              verbose=1).fit(X, y)

cart_final = cart_model.set_params(**cart_best_grid.best_params_, random_state=17).fit(X, y)

cv_results_6_1 = cross_validate(cart_final, X, y, cv=5, scoring=["accuracy", "f1", "roc_auc"])
cv_results_6_1["test_accuracy"].mean()
cv_results_6_1["test_f1"].mean()
cv_results_6_1["test_roc_auc"].mean()

# cart_final = accuracy= 0.8781, f1= 0.5842, roc_auc= 0.7225 / cart_model = accuracy= 0.7638, f1= 0.5428, roc_auc= 0.7309


#KNN

knn_model = KNeighborsClassifier()

cv_results_7 = cross_validate(knn_model, X, y, cv=5, scoring=["accuracy", "f1", "roc_auc"])
cv_results_7["test_accuracy"].mean()
cv_results_7["test_f1"].mean()
cv_results_7["test_roc_auc"].mean()

# accuracy= 0.5430, f1= 0.1800, roc_auc= 0.5192

#Hiperparametre optimizasyonu

knn_model.get_params()

knn_params = {"n_neighbors" : range(2, 50)}

knn_best_grid = GridSearchCV(knn_model,
                           knn_params,
                           cv=5,
                           n_jobs=-1,
                           verbose=1).fit(X, y)

knn_final = knn_model.set_params(**knn_best_grid.best_params_).fit(X, y)

cv_results_7_1 = cross_validate(knn_final, X, y, cv=5, scoring=["accuracy", "f1", "roc_auc"])
cv_results_7_1["test_accuracy"].mean()
cv_results_7_1["test_f1"].mean()
cv_results_7_1["test_roc_auc"].mean()

# knn_final = accuracy= 0.7934, f1= 0.0, roc_auc= 0.4771 / knn_model = accuracy= 0.5430, f1= 0.1800, roc_auc= 0.5192



#Logistic Regression

log_model = LogisticRegression(random_state=17)

cv_results_8 = cross_validate(log_model, X, y, cv=5, scoring=["accuracy", "f1", "roc_auc"])
cv_results_8["test_accuracy"].mean()
cv_results_8["test_f1"].mean()
cv_results_8["test_roc_auc"].mean()

# log_model = accuracy= 0.7934, f1= 0.0, roc_auc= 0.5543


# Final modeller cv_results sonuçları

# rf_final = accuracy= 0.8818, f1= 0.6148, roc_auc= 0.9096
# gbm_final = accuracy= 0.8858, f1= 0.6977, roc_auc= 0.8830
# xgboost_final = accuracy= 0.8819, f1= 0.6469, roc_auc= 0.8904
# lgbm_final = accuracy= 0.8856, f1= 0.6501, roc_auc= 0.8904
# catboost_final =  accuracy= 0.8781, f1= 0.5842, roc_auc= 0.8866
# cart_final = accuracy= 0.8781, f1= 0.5842, roc_auc= 0.7225
# knn_final = accuracy= 0.7934, f1= 0.0, roc_auc= 0.4771
# log_model = accuracy= 0.7934, f1= 0.0, roc_auc= 0.5543



#Feature importance

def plot_importance(model, features, num=len(X), save=False):
    feature_imp = pd.DataFrame({'Value': model.feature_importances_, 'Feature': features.columns})
    plt.figure(figsize=(10, 10))
    sns.set(font_scale=1)
    sns.barplot(x="Value", y="Feature", data=feature_imp.sort_values(by="Value",
                                                                     ascending=False)[0:num])
    plt.title('Features')
    plt.tight_layout()
    plt.show(block=True)
    if save:
        plt.savefig('importances.png')

plot_importance(rf_final, X)
plot_importance(xgboost_final, X)
plot_importance(lgbm_final, X)
plot_importance(catboost_final, X)
plot_importance(gbm_final, X)
plot_importance(cart_final, X)




















