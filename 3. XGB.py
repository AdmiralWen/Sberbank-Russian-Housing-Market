import os, sys
sys.path.append(os.path.expanduser('~') + '/Documents/Python/Custom Modules')
from DataScience import *

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

project_dir = os.path.expanduser('~') + "/OneDrive/Kaggle/Sberbank Russian Housing Market/Sberbank Russian Housing Market"
import xgboost as xgb

##########################################################
# Training Data Setup (Run Featuring Engineering First!) #

# Adjust extreme values of price per full_sq:
Sb_train_fe = Sb_train_fe[Sb_train_fe.price_doc/Sb_train_fe.full_sq_floored <= 600000]
Sb_train_fe = Sb_train_fe[Sb_train_fe.price_doc/Sb_train_fe.full_sq_floored >= 10000]
Sb_train_fe.shape

# Log the response variable:
Sb_train_fe['price_doc'] = Sb_train_fe['price_doc'].apply(lambda x: np.log1p(x))
Sb_train_fe['price_doc'].describe()

####################################
# Prepare Datasets for XGBoost Run #

test_id = Sb_test_fe['id'].copy()
train_target = Sb_train_fe['price_doc'].copy()
Sb_train_fe.drop(['id', 'timestamp', 'price_doc'], axis = 1, inplace = True)
Sb_test_fe.drop(['id', 'timestamp'], axis = 1, inplace = True)

# Effect code categorical variables:
Sb_train_fe_n = Sb_train_fe.select_dtypes(exclude=['object'])
Sb_train_fe_c = Sb_train_fe.select_dtypes(include=['object']).copy()
Sb_test_fe_n = Sb_test_fe.select_dtypes(exclude=['object'])
Sb_test_fe_c = Sb_test_fe.select_dtypes(include=['object']).copy()

for c in Sb_train_fe_c:
    Sb_train_fe_c[c] = pd.factorize(Sb_train_fe_c[c])[0]
for c in Sb_test_fe_c:
    Sb_test_fe_c[c] = pd.factorize(Sb_test_fe_c[c])[0]

Sb_train_fe = pd.concat([Sb_train_fe_n, Sb_train_fe_c], axis=1)
Sb_test_fe = pd.concat([Sb_test_fe_n, Sb_test_fe_c], axis=1)

# Create XGB matrices:
print(Sb_train_fe.shape, Sb_test_fe.shape)
Sb_train_DMat = xgb.DMatrix(Sb_train_fe.values, train_target.values, feature_names = Sb_train_fe.columns)
Sb_test_DMat = xgb.DMatrix(Sb_test_fe.values, feature_names = Sb_test_fe.columns)

#######################
# Initial XGBoost Run #

xgb_params = {
	'objective': 'reg:linear',
	'eval_metric': 'rmse',
	'eta': 0.05,
	'max_depth': 5,
	'subsample': 0.8,
	'colsample_bytree': 0.7,
	"min_child_weight": 1,
	"gamma": 0,
	"alpha": 0,
	"nthread": 8
}

test_model = xgb.train(params = xgb_params, dtrain = Sb_train_DMat, num_boost_round = 750)
test_model.best_iteration

plt.rcParams.update({'font.size': 16})
fig, ax = plt.subplots(1, 1, figsize=(15, 15))
xgb.plot_importance(test_model, max_num_features = 50, height = 0.5, ax = ax)
plt.show()

############################
# XGBoost Cross-Validation #

xgb_cv_params = {
	'objective': 'reg:linear',
	'eval_metric': 'rmse',
	'eta': 0.05,
	'max_depth': 5,
	'subsample': 0.8,
	'colsample_bytree': 0.7,
	"min_child_weight": 1,
	"gamma": 0,
	"alpha": 0,
	"nthread": 8
}

# Tune max_depth:
for i in range(4, 8):
	xgb_cv_params["max_depth"] = i
	xgb_cv1 = xgb.cv(params = xgb_cv_params, dtrain = Sb_train_DMat, nfold = 6, num_boost_round = 1000, early_stopping_rounds = 25,
					 verbose_eval = False, seed = 1990)
	print("Depth=", i, ":", float(xgb_cv1['test-rmse-mean'][-1:]))
xgb_cv_params["max_depth"] = 4

# Tune subsample and colsample_bytree:
for i in np.arange(0.5, 1.01, 0.1):
	xgb_cv_params["subsample"] = i
	for j in np.arange(0.3, 1.01, 0.1):
		xgb_cv_params['colsample_bytree'] = j
		xgb_cv1 = xgb.cv(params = xgb_cv_params, dtrain = Sb_train_DMat, nfold = 6, num_boost_round = 1000, early_stopping_rounds = 25,
						 verbose_eval = False, seed = 1990)
		print("Subsample=", i, ", Colsample=", j, ":", float(xgb_cv1['test-rmse-mean'][-1:]))
xgb_cv_params["subsample"] = 0.9
xgb_cv_params["colsample_bytree"] = 0.9

# Tune min_child_weight:
for i in [1, 2, 5, 8, 10, 15, 20]:
	xgb_cv_params["min_child_weight"] = i
	xgb_cv1 = xgb.cv(params = xgb_cv_params, dtrain = Sb_train_DMat, nfold = 6, num_boost_round = 1000, early_stopping_rounds = 25,
					 verbose_eval = False, seed = 1990)
	print("MinChildWt=", i, ":", float(xgb_cv1['test-rmse-mean'][-1:]))
xgb_cv_params["min_child_weight"] = 10

# Tune gamma:
for i in np.arange(0, 1.01, 0.25):
	xgb_cv_params["gamma"] = i
	xgb_cv1 = xgb.cv(params = xgb_cv_params, dtrain = Sb_train_DMat, nfold = 6, num_boost_round = 1000, early_stopping_rounds = 25,
					 verbose_eval = False, seed = 1990)
	print("Gamma=", i, ":", float(xgb_cv1['test-rmse-mean'][-1:]))
xgb_cv_params["gamma"] = 1

# Tune alpha:
for i in np.arange(0, 2.01, 0.5):
	xgb_cv_params["alpha"] = i
	xgb_cv1 = xgb.cv(params = xgb_cv_params, dtrain = Sb_train_DMat, nfold = 6, num_boost_round = 1000, early_stopping_rounds = 25,
					 verbose_eval = False, seed = 1990)
	print("Alpha=", i, ":", float(xgb_cv1['test-rmse-mean'][-1:]))
xgb_cv_params["alpha"] = 1.5

####################################
# XGBoost Cross-Validation Round 2 #

xgb_cv_params2 = {
	'objective': 'reg:linear',
	'eval_metric': 'rmse',
	'eta': 0.02,
	'max_depth': 4,
	'subsample': 0.9,
	'colsample_bytree': 0.9,
	"min_child_weight": 10,
	"gamma": 1,
	"alpha": 1.5,
	"nthread": 8
}

xgb_cv2 = xgb.cv(params = xgb_cv_params2, dtrain = Sb_train_DMat, nfold = 8, num_boost_round = 1500, early_stopping_rounds = 50,
					verbose_eval = True, seed = 1990)

###################
# Fit Final Model #

# Model fit run:
xgb_fit1 = xgb.train(params = xgb_cv_params2, dtrain = Sb_train_DMat, num_boost_round = 1039)

# Plot variable importance graph:
plt.rcParams.update({'font.size': 16})
fig, ax = plt.subplots(1, 1, figsize=(15, 15))
xgb.plot_importance(xgb_fit1, max_num_features = 50, height = 0.5, ax = ax)
plt.show()

# Visualize the first tree in the boosted sequence (note that graphviz must be installed):
xgb.plot_tree(xgb_fit1, num_trees = 0)
plt.show()

###########################
# Predict on Test Dataset #

price_pred_log = xgb_fit1.predict(Sb_test_DMat)
price_pred = np.exp(price_pred_log) - 1

xgb_sub1 = pd.DataFrame({'id':test_id, 'price_doc':price_pred})
xgb_sub1.to_csv(project_dir + "/Submissions/xgb_sub1.csv", index = False)