# Databricks notebook source
import pandas as pd
import matplotlib.pyplot as plt
import xgboost as xgb
import numpy as np

# COMMAND ----------

data_path = '/dbfs/FileStore/ChallengeWindPower/'
raw_data = pd.read_csv(f'{data_path}train_phase_1.csv')
raw_data.date = pd.to_datetime(raw_data.date, format='%Y-%m-%d %H:%M:%S')
raw_test = pd.read_csv(f'{data_path}test_phase_1.csv')
raw_test.date = pd.to_datetime(raw_test.date, format='%Y-%m-%d %H:%M:%S')
data=pd.concat([raw_data,raw_test])
data.sort_values(by=['date'],inplace=True)
data.reset_index(inplace=True,drop=True)
test_ind=data[data.wp1.isna()].index
train_ind=data[data.wp1.notna()].index

# COMMAND ----------

for i in range(0,84):
  data.loc[i::84,'window_position']=i+1
for i in range(1,len(data)//84+2):
  data.loc[(i-1)*84:i*84,'window']=i

# COMMAND ----------

# 5 folds

myCViterator = []
windows=list(range(1,len(data)//84+2))

for i in range(1,6):
  print(len(windows))
  val_windows=list(range(i,len(data)//84+2,5))
  train_windows=[wind for wind in windows if wind not in val_windows]
    
  train_ind=data.loc[data.window.isin(train_windows)].index
  val_ind=data.loc[data.window.isin(val_windows)].index

  myCViterator.append((train_ind, val_ind))

# COMMAND ----------

import datetime as dt
data['hour']=data.date.dt.hour
data['month']=data.date.dt.month
data['year']=data.date.dt.year

# COMMAND ----------

data['wd_cut']=pd.cut(data.wd,8)

# COMMAND ----------

train_lr=data.dropna(subset=['wp1'])

# COMMAND ----------

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, PolynomialFeatures

full_pipeline=ColumnTransformer([
  ('cat',OneHotEncoder(sparse=False),['wd_cut']),
  ('poly',PolynomialFeatures(degree=4,include_bias=False),['ws'])
])

X_train=full_pipeline.fit_transform(train_lr)
y_train=train_lr.wp1

# COMMAND ----------

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
lm = LinearRegression()
lm.fit(X_train, y_train)

# COMMAND ----------

data['proxy_wp1']=lm.predict(full_pipeline.fit_transform(data))

# COMMAND ----------

window_size=3
for i in range(1,window_size+1):
  data['proxy_wp1.p'+str(i)]=data['proxy_wp1'].shift(i)
  data['proxy_wp1.n'+str(i)]=data['proxy_wp1'].shift(-i)

# COMMAND ----------

# clustering
data_split=np.split(data['ws.angle'],len(data)/12)
from sklearn.cluster import KMeans
kmeans=KMeans(n_clusters=8)
clusters=kmeans.fit_predict(data_split)
data['cluster']=0
for i, cluster in enumerate(clusters):
  data.loc[i*12:12*(i+1)-1,'cluster']=cluster
  data.loc[i*12:12*(i+1)-1,'begin']=data.loc[i*12,'hour']
data['begin']=data.begin.astype(int)

# COMMAND ----------

data['proxy_wp1.n1'].fillna(method='ffill',inplace=True)
data['proxy_wp1.n2'].fillna(method='ffill',inplace=True)
data['proxy_wp1.n3'].fillna(method='ffill',inplace=True)

# COMMAND ----------

data['proxy_wp1.p1'].fillna(method='bfill',inplace=True)
data['proxy_wp1.p2'].fillna(method='bfill',inplace=True)
data['proxy_wp1.p3'].fillna(method='bfill',inplace=True)

# COMMAND ----------

data.wp1.fillna(data['proxy_wp1'],inplace=True)

# COMMAND ----------

X_train=data.loc[:,['ws','proxy_wp1','proxy_wp1.p1', 'proxy_wp1.n1', 'proxy_wp1.p2', 'proxy_wp1.n2',
       'proxy_wp1.p3', 'proxy_wp1.n3','hour','month','year']]

y_train=data.loc[:,'wp1']

# COMMAND ----------

from sklearn.model_selection import cross_val_score
from sklearn.metrics import make_scorer, r2_score
from numpy import mean
from hyperopt import STATUS_OK
from sklearn.ensemble import GradientBoostingRegressor
  
def objective_function(params):
  
 # set the hyperparameters that we want to tune:
  n_estimators = params["n_estimators"]
  max_depth = params["max_depth"]
  max_features = params["max_features"]
  min_samples_split = params["min_samples_split"]
  min_samples_leaf = params["min_samples_leaf"]
  learning_rate = params["learning_rate"]
  

  gbr = GradientBoostingRegressor(n_estimators=n_estimators,
                             max_depth=max_depth,
                             max_features=max_features,
                             min_samples_split=min_samples_split,
                             min_samples_leaf=min_samples_leaf,
                             learning_rate=learning_rate,
                             random_state=42)
  # Evaluate predictions
  nmae = mean(cross_val_score(gbr, X_train, y_train, scoring='neg_mean_absolute_error',cv=myCViterator))
#   reg1.fit(X_train,y_train)
#   mae=mean_absolute_error(reg1.predict(X_val), y_val)


  # Note: since we aim to maximize r2, we need to return it as a negative value ("loss": -metric)
  return {"loss": -nmae, "status": STATUS_OK}

# COMMAND ----------

from hyperopt import hp

max_features=["auto", "sqrt", "log2"]

search_space = {
  "n_estimators":hp.randint('n_estimators',30,2000),
  "max_depth": hp.randint("max_depth", 1, 10),
  "min_samples_split": hp.uniform("min_samples_split", 0, 1),
  "min_samples_leaf": hp.randint("min_samples_leaf", 1, 10),
  "max_features": hp.choice("max_features", max_features),
  'learning_rate': hp.uniform("learning_rate",0,1)
  
}

# COMMAND ----------

# fine-tuning model 1

from hyperopt import fmin, tpe, STATUS_OK, SparkTrials
import mlflow

# Creating a parent run
with mlflow.start_run():
  # the number of models we want to evaluate
  num_evals = 20
  # set the number of models to be trained concurrently
  spark_trials = SparkTrials(parallelism=2)
  best_hyperparam = fmin(fn = objective_function, 
                         space = search_space,
                         algo = tpe.suggest, 
                         trials = spark_trials,
                         max_evals = num_evals)

  # get optimal hyperparameter values
  best_n_estimators=best_hyperparam["n_estimators"]
  best_max_depth = best_hyperparam["max_depth"]
#   best_gamma = best_hyperparam["gamma"]
#   best_reg_alpha = best_hyperparam["reg_alpha"]
#   best_reg_lambda = best_hyperparam["reg_lambda"]
#   best_colsample_bytree = best_hyperparam["colsample_bytree"]
  best_learning_rate = best_hyperparam["learning_rate"]
#   best_min_child_weight=best_hyperparam["min_child_weight"]
  best_max_features=max_features[best_hyperparam["max_features"]]
  best_min_samples_split=best_hyperparam["min_samples_split"]
  best_min_samples_leaf=best_hyperparam["min_samples_leaf"]
  
  mlflow.log_param('best_n_estimators',best_n_estimators)
  mlflow.log_param('best_max_depth',best_max_depth)
  mlflow.log_param('best_learning_rate',best_learning_rate)
  mlflow.log_param('best_max_features',best_max_features)
  mlflow.log_param('best_min_samples_split',best_min_samples_split)
  mlflow.log_param('best_min_samples_leaf',best_min_samples_leaf)
  
  reg1 = GradientBoostingRegressor(n_estimators=best_n_estimators,
                             max_depth=best_max_depth,
                             max_features=best_max_features,
                             min_samples_split=best_min_samples_split,
                             min_samples_leaf=best_min_samples_leaf,
                             learning_rate=best_learning_rate,
                             random_state=42)
  
  mlflow.sklearn.log_model(reg1,'best reg1')

  # train model on entire training data
 

# COMMAND ----------

X_train=data.loc[:,['proxy_wp1.p1','proxy_wp1.p2','proxy_wp1.p3','hour','month','cluster','begin']]
y_train=data.loc[:,'wp1']

# COMMAND ----------

# fine-tuning model 2

with mlflow.start_run(run_name='best_reg3'):
  # the number of models we want to evaluate
  num_evals = 20
  # set the number of models to be trained concurrently
  spark_trials = SparkTrials(parallelism=2)
  best_hyperparam = fmin(fn = objective_function, 
                         space = search_space,
                         algo = tpe.suggest, 
                         trials = spark_trials,
                         max_evals = num_evals)

  # get optimal hyperparameter values
  best_n_estimators=best_hyperparam["n_estimators"]
  best_max_depth = best_hyperparam["max_depth"]
  best_max_features=max_features[best_hyperparam["max_features"]]
  best_min_samples_split=best_hyperparam["min_samples_split"]
  best_min_samples_leaf=best_hyperparam["min_samples_leaf"]
  best_learning_rate=best_hyperparam["learning_rate"]
  
  mlflow.log_param('best_n_estimators',best_n_estimators)
  mlflow.log_param('best_max_depth',best_max_depth)
  mlflow.log_param('best_learning_rate',best_learning_rate)
  mlflow.log_param('best_max_features',best_max_features)
  mlflow.log_param('best_min_samples_split',best_min_samples_split)
  mlflow.log_param('best_min_samples_leaf',best_min_samples_leaf)

  # train model on entire training data
  reg3 = GradientBoostingRegressor(n_estimators=best_n_estimators,
                             max_depth=best_max_depth, 
                             max_features=best_max_features, 
                             min_samples_split=best_min_samples_split,
                             min_samples_leaf=best_min_samples_leaf,
                             learning_rate=best_learning_rate,
                             random_state=42)
  mlflow.sklearn.log_model(reg3,'best reg3')

# COMMAND ----------

# Training all the models on the whole DF


X_train=data.loc[train_ind,['ws','proxy_wp1','proxy_wp1.p1', 'proxy_wp1.n1', 'proxy_wp1.p2', 'proxy_wp1.n2',
       'proxy_wp1.p3', 'proxy_wp1.n3','hour','month','year']]
y_train=data.loc[train_ind,'wp1']

reg1.fit(X_train, y_train)

X_pred=data.loc[:,['ws','proxy_wp1','proxy_wp1.p1', 'proxy_wp1.n1', 'proxy_wp1.p2', 'proxy_wp1.n2',
       'proxy_wp1.p3', 'proxy_wp1.n3','hour','month','year']]

data['predictions_gbm1']=reg1.predict(X_pred)


X_train=data.loc[train_ind,['proxy_wp1.p1','proxy_wp1.p2','proxy_wp1.p3','hour','month','cluster','begin']]
y_train=data.loc[train_ind,'wp1']

reg3.fit(X_train, y_train)

X_pred=data.loc[:,['proxy_wp1.p1','proxy_wp1.p2','proxy_wp1.p3','hour','month','cluster','begin']]

data['predictions_gbm3']=reg3.predict(X_pred)


# COMMAND ----------

X_train=data.loc[train_ind,['predictions_gbm1','predictions_gbm3']]
y_train=data.loc[train_ind,'wp1']

lm_ens = LinearRegression()
lm_ens.fit(X_train, y_train)

X_pred=data.loc[:,['predictions_gbm1','predictions_gbm3']]
data['prediction_ens']=lm_ens.predict(X_pred)

# COMMAND ----------

predictions=data.loc[test_ind,['date','prediction_ens']]

# COMMAND ----------

predictions.rename(columns={'prediction_ens':'wp1'},inplace=True)
predictions.reset_index(inplace=True,drop=True)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Correction of the first and last values of each 48h window with auto-regressive models

# COMMAND ----------

all_data=pd.concat([raw_data,raw_test])
all_data.sort_values(by=['date'],inplace=True)
all_data.reset_index(inplace=True,drop=True)
data=all_data.copy()
data.set_index('date',inplace=True)
data['month'] = data.index.month
data['weekday'] = data.index.weekday
data['hour']=data.index.hour
data['year']=data.index.year
data['day']=data.index.day
data['doy']=data.index.dayofyear
data['dim']=data.index.days_in_month
first_hour=data.index[0] #careful
data['hours_s_begin']=(data.index-first_hour).astype('timedelta64[h]')
data['sin_month']=np.sin(2.*np.pi*data.month/12)
data['cos_month']=np.cos(2.*np.pi*data.month/12)
data['sin_weekday']=np.sin(2.*np.pi*data.weekday/7)
data['cos_weekday']=np.cos(2.*np.pi*data.weekday/7)
data['sin_wd']=np.sin(2.*np.pi*data.wd/360)
data['cos_wd']=np.cos(2.*np.pi*data.wd/360)
data['sin_doy']=np.sin(2.*np.pi*data.doy/365)
data['cos_doy']=np.cos(2.*np.pi*data.doy/365)
data['sin_hour']=np.sin(2.*np.pi*data.hour/24)
data['cos_hour']=np.cos(2.*np.pi*data.hour/24)
data['sin_day']=np.sin(2.*np.pi*data.day/data.dim)
data['cos_day']=np.cos(2.*np.pi*data.day/data.dim)
data.drop(['month','weekday','wd','doy','hour','day','dim'],axis=1,inplace=True)

# COMMAND ----------

window_size=5
for i in range(1,window_size):
  data['wp-t-'+str(i)]=data.wp1.shift(i)

# COMMAND ----------

train=data.dropna(axis=0)

# COMMAND ----------

predictions.set_index('date',inplace=True)

# COMMAND ----------

from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error

X_train = train.drop(['wp1'],axis=1).values
y_train = train['wp1'].values
rf_lag = GradientBoostingRegressor()
rf_lag.fit(X_train, y_train)

for ind in predictions.index[0::48]:
  X_new=np.array(data.drop(['wp1'],axis=1).loc[ind,:])[:,np.newaxis]
  X_new=np.transpose(X_new)
  pred=rf_lag_0.predict(X_new)
  predictions.loc[ind,'wp1']=pred
  
X_train = train.drop(['wp1','wp-t-1'],axis=1).values
y_train = train['wp1'].values
rf_lag_1 = GradientBoostingRegressor()
rf_lag_1.fit(X_train, y_train)

for ind in predictions.index[1::48]:
  X_new=np.array(data.drop(['wp1','wp-t-1'],axis=1).loc[ind,:])[:,np.newaxis]
  X_new=np.transpose(X_new)
  pred=rf_lag_1.predict(X_new)
  predictions.loc[ind,'wp1']=pred


# COMMAND ----------

data=all_data.copy()
data.set_index('date',inplace=True)
data['month'] = data.index.month
data['weekday'] = data.index.weekday
data['hour']=data.index.hour
data['year']=data.index.year
data['day']=data.index.day
data['doy']=data.index.dayofyear
data['dim']=data.index.days_in_month
first_hour=data.index[0] #careful
data['hours_s_begin']=(data.index-first_hour).astype('timedelta64[h]')
data['sin_month']=np.sin(2.*np.pi*data.month/12)
data['cos_month']=np.cos(2.*np.pi*data.month/12)
data['sin_weekday']=np.sin(2.*np.pi*data.weekday/7)
data['cos_weekday']=np.cos(2.*np.pi*data.weekday/7)
data['sin_wd']=np.sin(2.*np.pi*data.wd/360)
data['cos_wd']=np.cos(2.*np.pi*data.wd/360)
data['sin_doy']=np.sin(2.*np.pi*data.doy/365)
data['cos_doy']=np.cos(2.*np.pi*data.doy/365)
data['sin_hour']=np.sin(2.*np.pi*data.hour/24)
data['cos_hour']=np.cos(2.*np.pi*data.hour/24)
data['sin_day']=np.sin(2.*np.pi*data.day/data.dim)
data['cos_day']=np.cos(2.*np.pi*data.day/data.dim)
data.drop(['month','weekday','wd','doy','hour','day','dim'],axis=1,inplace=True)

# COMMAND ----------

window_size=5
for i in range(1,window_size):
  data['wp-t+'+str(i)]=data.wp1.shift(-i)

# COMMAND ----------

X_train = train.drop(['wp1'],axis=1).values
y_train = train['wp1'].values
rf_adv = GradientBoostingRegressor()
rf_adv.fit(X_train, y_train)

for ind in predictions.index[47::48]:
  X_new=np.array(data.drop(['wp1'],axis=1).loc[ind,:])[:,np.newaxis]
  X_new=np.transpose(X_new)
  pred=rf_adv.predict(X_new)
  predictions.loc[ind,'wp1']=pred
  
X_train = train.drop(['wp1','wp-t+1'],axis=1).values
y_train = train['wp1'].values
rf_adv_1 = GradientBoostingRegressor()
rf_adv_1.fit(X_train, y_train)

for ind in predictions.index[46::48]:
  X_new=np.array(data.drop(['wp1','wp-t+1'],axis=1).loc[ind,:])[:,np.newaxis]
  X_new=np.transpose(X_new)
  pred=rf_adv_1.predict(X_new)
  predictions.loc[ind,'wp1']=pred

# COMMAND ----------

predictions.loc[predictions.wp1<0,'wp1']=0
predictions.reset_index(inplace=True)
predictions.to_csv('/dbfs/FileStore/ChallengeWindPower/predictions.csv', index=False, sep=';')