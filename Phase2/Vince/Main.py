# Databricks notebook source
# Import Data
import pandas as pd

data_path = '/dbfs/FileStore/ChallengeWindPower/Phase2/'
train=pd.read_csv(f'{data_path}train.csv')
test=pd.read_csv(f'{data_path}test.csv')
wf1=pd.read_csv(f'{data_path}wp1.csv')
wf2=pd.read_csv(f'{data_path}wp2.csv')
wf3=pd.read_csv(f'{data_path}wp3.csv')
wf4=pd.read_csv(f'{data_path}wp4.csv')
wf5=pd.read_csv(f'{data_path}wp5.csv')
wf6=pd.read_csv(f'{data_path}wp6.csv')
data=pd.concat([train,test])
data.sort_values(by=['date'],inplace=True)
data.reset_index(inplace=True,drop=True)

# COMMAND ----------

# Here we're doing some data augmentation setting each farm wp as an instance to train on

# pivot
train_pivot=(data
 .melt(id_vars='date',var_name='farm',value_name='wp')
 .sort_values(['date','farm'])
 .reset_index(drop=True)
)
# wpX-->X
train_pivot['farm']=train_pivot['farm'].str[-1].astype(int)
train_pivot['date']=pd.to_datetime(train_pivot.date,format='%Y%m%d%H')

train_pivot['hour']=train_pivot['date'].dt.hour


# COMMAND ----------

# concat wind forecasts
from datetime import timedelta
wf1['farm']=1
wf2['farm']=2
wf3['farm']=3
wf4['farm']=4
wf5['farm']=5
wf6['farm']=6
wind_forecasts_df=pd.concat([wf1,wf2,wf3,wf4,wf5,wf6])
wind_forecasts_df['start']=pd.to_datetime(wind_forecasts_df['date'],format='%Y%m%d%H')
wind_forecasts_df['start_hour']=wind_forecasts_df['start'].dt.hour
wind_forecasts_df['date']=pd.to_datetime(wind_forecasts_df['date'],format='%Y%m%d%H')+(wind_forecasts_df['hors']/24.0).map(timedelta)
wind_forecasts_df['wd_cut']=pd.cut(wind_forecasts_df['wd'],8)

# COMMAND ----------

# Creation of a custom feature with a simple linear regression

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, PolynomialFeatures
from sklearn.linear_model import LinearRegression

ws_angle_df=pd.merge(wind_forecasts_df,train_pivot,how='left',left_on=['date','farm'],right_on=['date','farm'])
ws_angle_df.dropna(subset=['u'],inplace=True) # we don't have always 4 forecasts for one date

full_pipeline=ColumnTransformer([
  ('cat',OneHotEncoder(sparse=False),['wd_cut']),
  ('poly',PolynomialFeatures(degree=4,include_bias=False),['ws'])
])

train_lr=ws_angle_df.set_index('date').loc[:'2010'] #we will be able to do some incremental training here

X_train=full_pipeline.fit_transform(train_lr)
y_train=train_lr.wp

lm = LinearRegression()
lm.fit(X_train, y_train)

ws_angle_df['ws_angle']=lm.predict(full_pipeline.fit_transform(ws_angle_df))

# COMMAND ----------

ws_angle_df=ws_angle_df[['date','farm','hors','wp','ws_angle','ws']].rename(columns={'hors':'dist'})

window_size=3
for i in range(1,window_size+1):
  ws_angle_df['ws_angle_p'+str(i)]=ws_angle_df['ws_angle'].shift(i)
  ws_angle_df['ws_angle_p'+str(i)].fillna(method='bfill',inplace=True)
  ws_angle_df['ws_angle_n'+str(i)]=ws_angle_df['ws_angle'].shift(-i)
  ws_angle_df['ws_angle_n'+str(i)].fillna(method='ffill',inplace=True)

# COMMAND ----------

# forecasts_and_wp_df[pd.isnull(forecasts_and_wp_df).any(axis=1)]

# COMMAND ----------

# clustering per farm

from sklearn.cluster import KMeans
import numpy as np

clusters_df=pd.DataFrame()

for farm in range(1,7):
  vect=ws_angle_df.loc[ws_angle_df['farm']==farm].set_index('date')['ws_angle']
  X_tot=np.split(vect,len(vect)/12) #checked that it is always the same size
#   We can't use future data for training
  X_train=[window for window in X_tot if window.index[11].year<=2010] #incremental training
#   training and predictions
  kmeans=KMeans(n_clusters=6)
  kmeans.fit(X_train)
  clusters=kmeans.predict(X_tot)
# Filling of the dataset
  results_df=ws_angle_df[['date','farm','dist']].loc[ws_angle_df.farm==farm]
  results_df['hour']=results_df['date'].dt.hour
  results_df['farm_cluster']=0
  results_df['window_start']=0
  results_df['window_instance_pos']=0
  for i, cluster in enumerate(clusters):
    results_df['window_instance_pos'].iloc[i*12:12*(i+1)]=range(1,13) #checké si on a le même entre un cluster par farm et un cluster total
    results_df['farm_cluster'].iloc[i*12:12*(i+1)]=cluster
    results_df['window_start'].iloc[i*12:12*(i+1)]=results_df['hour'].iloc[i*12]
  results_df['window_start']=results_df.window_start.astype(int)
#   Concatenation
  clusters_df=pd.concat([clusters_df,results_df])
  clusters_df.drop(['hour'],axis=1)

# COMMAND ----------

# overall clustering
vect=ws_angle_df.set_index('date')['ws_angle']
X_tot=np.split(vect,len(vect)/12) #1 farm per window
#   We can't use future data for training
X_train=[window for window in X_tot if window.index[11].year<=2010]
#   training and predictions
kmeans=KMeans(n_clusters=24)
kmeans.fit(X_train)
clusters=kmeans.predict(X_tot)
# Filling of the dataset
clusters_df['general_cluster']=0
for i, cluster in enumerate(clusters):
  clusters_df['general_cluster'].iloc[i*12:12*(i+1)]=cluster

# COMMAND ----------

# Join everything
import datetime as dt

prepared_df=pd.merge(ws_angle_df,clusters_df,how='left',left_on=['date','farm','dist'],right_on=['date','farm','dist'])

prepared_df['month']=prepared_df['date'].dt.month
prepared_df['year']=prepared_df['date'].dt.year

# COMMAND ----------

def get_features_and_targets(
                         df:pd.DataFrame,
                         features_list:list,
                         sample:bool,
                         farm:int=None,
                         dist:range=None) -> (pd.DataFrame, pd.DataFrame):
  
  if farm is not None:
    filtered_df=df[df['farm']==farm]
  elif dist is not None:
    mask=df['dist'].isin(dist)
    filtered_df=df.loc[mask]
  else:
    filtered_df=df
    
  X_train=filtered_df.loc[sample,features_list] 
  y_train=filtered_df.loc[sample,'wp']
  
  return (X_train, y_train)

# COMMAND ----------

predictions_df=prepared_df[['date','farm','dist']].copy()

# COMMAND ----------

import xgboost as xgb

def model_training(X_train, y_train, params, trained_model=None ):
    xg_train = xgb.DMatrix(X_train, label=y_train)
    if trained_model is None: #we don't use a pretrain model
      model=xgb.train(params=params, dtrain=xg_train)
    else:
      model=xgb.train(params=params, dtrain=xg_train, xgb_model=trained_model)
      print('use of a trained model')
    return model

def get_predictions(X_pred, model):
    xg_pred=xgb.DMatrix(X_pred)
    y_pred=model.predict(xg_pred)
    return y_pred
  
def train_predict_and_save_model(train_df, features, training_sample, testing_sample, pred_df, model, params, **kwargs):
  if 'farm' in kwargs.keys():
    farm=kwargs['farm']
    X_train, y_train = get_features_and_targets(train_df, features, training_sample, farm=farm ) 
    X_pred, _ = get_features_and_targets(train_df, features, testing_sample, farm=farm )
    predictions_mask=(pred_df['farm']==farm) & testing_sample
#     model_name = model + '_farm_' + str(farm) #on ne distingue pas model1 et model2 --> pas besoin, ils sont écrasés quand on passe au modèle2
    predictions_column='predictions_'+model+'_farm'
    print(f"training on farm {farm}")
  elif 'dist' in kwargs.keys():
    dist=kwargs['dist']
    X_train, y_train = get_features_and_targets(train_df, features, training_sample, dist=dist ) 
    X_pred, _ = get_features_and_targets(train_df, features, testing_sample, dist=dist )
    mask=pred_df['dist'].isin(dist)
    predictions_mask=mask & testing_sample
#     model_name= model + '_dist_' + str(dist)
    predictions_column='predictions_'+model+'_dist'
    print(f"training on dist {dist}")
  else:
    X_train, y_train = get_features_and_targets(train_df, features, training_sample ) 
    X_pred, _ = get_features_and_targets(train_df, features, testing_sample )
    predictions_mask=testing_sample
#     model_name = model + '_all'
    predictions_column = 'predictions_'+model+'_all'
    print('overall training')
  
  if 'trained_model' in kwargs.keys():
    trained_model=kwargs['trained_model']
    model=model_training(X_train, y_train, params, trained_model=trained_model)
  else:
    model=model_training(X_train, y_train, params)
    
  y_pred = get_predictions(X_pred, model)
  pred_df.loc[predictions_mask,predictions_column]=y_pred
#   model.save_model(model_name)
  
def train_sample(df, window_start, window_end):
  return ((df['date']>=window_start) & (df['date']<=window_end)) | ((df['date']>=BEGINNING_TRAIN_PERIOD) & (df['date']<=END_TRAIN_PERIOD))

def write_sample(df, window_start, window_end):
  return (df['date']>=window_start) & (df['date']<=window_end)

def first_samples(df, window_size=47):
  training_sample = train_sample(df, BEGINNING_TRAIN_PERIOD, END_TRAIN_PERIOD)
  test_window_end = BEGINNING_TEST_PERIOD+dt.timedelta(hours=window_size)
  testing_sample = write_sample(df, BEGINNING_TRAIN_PERIOD, test_window_end)
  return training_sample, testing_sample

def initialization(test_window_size=48, train_window_size=36):
  test_window_start = BEGINNING_TEST_PERIOD+timedelta(hours=test_window_size+train_window_size)
  train_window_start = BEGINNING_TEST_PERIOD+dt.timedelta(hours=test_window_size)
  return train_window_start, test_window_start

# COMMAND ----------

features_1=['ws','farm','dist','ws_angle','ws_angle_p1', 'ws_angle_n1', 'ws_angle_p2', 'ws_angle_n2',
       'ws_angle_p3', 'ws_angle_n3','hour','month','year']

params=  {'colsample_bytree': 0.6,
   'max_depth': 9,
   'min_child_weight': 5,
   'eval_metric': 'mae',
   'subsample': 0.6,
          
#    'colsample': 1.0,
   'eta': 0.05}
  
# first training until 2010

BEGINNING_TEST_PERIOD=dt.datetime(2011,1,1,1)
END_TRAIN_PERIOD=dt.datetime(2011,1,1,0)
BEGINNING_TRAIN_PERIOD=dt.datetime(2009,7,1,1)
END_TEST_PERIOD=dt.datetime(2012,6,23,1)

training_sample, testing_sample = first_samples(prepared_df)

for farm in range(1,7):
  train_predict_and_save_model(train_df=prepared_df, features=features_1, training_sample=training_sample, testing_sample=testing_sample, pred_df=predictions_df, model='model1', params=params, farm=farm)
  
n = 3 # chunk length
distances=range(1,49)
chunks = [distances[i:i+n] for i in range(0, len(distances), n)]

for dist in chunks:
  train_predict_and_save_model(train_df=prepared_df, features=features_1, training_sample=training_sample, testing_sample=testing_sample, pred_df=predictions_df, model='model1', params=params, dist=dist)

train_predict_and_save_model(train_df=prepared_df, features=features_1, training_sample=training_sample, testing_sample=testing_sample, pred_df=predictions_df,  model='model1', params=params)

# COMMAND ----------

#incremental training

# per farm

params=  {
  'colsample_bytree': 0.6,
  'max_depth': 9,
  'min_child_weight': 5,
  'eval_metric': 'mae',
  'subsample': 0.6,
#   'updater':'refresh',
#   'process_type': 'update',
#   'refresh_leaf': True,
#   'silent': False,
#    'colsample': 1.0,
   'eta': 0.05}


for farm in range(1,7):
  
  train_window_start, test_window_start = initialization()

  while test_window_start<=END_TEST_PERIOD:
    
    test_window_end = test_window_start+dt.timedelta(hours=47)
    train_window_end = train_window_start+dt.timedelta(hours=35)
    
    training_sample = train_sample(prepared_df, train_window_start, train_window_end)
    testing_sample = write_sample(prepared_df, train_window_start, test_window_end)
    
#     print(f" training from {train_window_start} and {train_window_end}")
#     print(f" writing from {train_window_start} and {test_window_end}")
#     print("-------------------------------------------------------------")
    
#     trained_model='model1_farm_' + str(farm)

    train_predict_and_save_model(train_df=prepared_df, 
                                 features=features_1, 
                                 training_sample=training_sample, 
                                 testing_sample=testing_sample, 
                                 pred_df=predictions_df,  
                                 model='model1', 
                                 params=params, farm=farm 
#                                  trained_model=trained_model
                                )

    test_window_start=test_window_end+dt.timedelta(hours=37)
    train_window_start=train_window_end+dt.timedelta(hours=49)
    

# COMMAND ----------

# per dist

for dist in chunks:
  
  train_window_start, test_window_start = initialization()

  while test_window_start<=END_TEST_PERIOD:
    
    test_window_end=test_window_start+dt.timedelta(hours=47)
    train_window_end=train_window_start+dt.timedelta(hours=35)
    
    training_sample = train_sample(prepared_df, train_window_start, train_window_end)
    testing_sample = write_sample(prepared_df, train_window_start, test_window_end)
#     trained_model='model1_dist_' + str(dist)
    train_predict_and_save_model(train_df=prepared_df,
                                 features=features_1,
                                 training_sample=training_sample,
                                 testing_sample=testing_sample,
                                 pred_df=predictions_df, 
                                 model='model1', params=params, 
                                 dist=dist
#                                  ,trained_model= trained_model
                                )

    test_window_start=test_window_end+dt.timedelta(hours=37)
    train_window_start=train_window_end+dt.timedelta(hours=49)

# overall

train_window_start, test_window_start = initialization()
  
while test_window_start<=END_TEST_PERIOD:
  
  test_window_end=test_window_start+dt.timedelta(hours=47)
  train_window_end=train_window_start+dt.timedelta(hours=35)
  training_sample = train_sample(prepared_df, train_window_start, train_window_end)
  testing_sample = write_sample(prepared_df, train_window_start, test_window_end)
#   trained_model = 'model1_all'
  train_predict_and_save_model(train_df=prepared_df,
                               features=features_1, 
                               training_sample=training_sample, 
                               testing_sample=testing_sample, 
                               pred_df=predictions_df, 
                               model='model1', 
                               params=params 
#                                trained_model=trained_model
                              )

  test_window_start=test_window_end+dt.timedelta(hours=37)
  train_window_start=train_window_end+dt.timedelta(hours=49)

# COMMAND ----------

features_2=['farm','dist','ws_angle_p1', 'ws_angle_p2',
       'ws_angle_p3','hour','month','farm_cluster','general_cluster','window_start','window_instance_pos']

params=  {'colsample_bytree': 0.6,
   'max_depth': 9,
   'min_child_weight': 5,
   'eval_metric': 'mae',
   'subsample': 0.6,
#    'colsample': 1.0,
   'eta': 0.05}
  
# first training until 2010

training_sample, testing_sample = first_samples(prepared_df)

for farm in range(1,7):
  train_predict_and_save_model(train_df=prepared_df, features=features_2, training_sample=training_sample, testing_sample=testing_sample, pred_df=predictions_df, model='model2', params=params, farm=farm)
  
n = 3 # chunk length
distances=range(1,49)
chunks = [distances[i:i+n] for i in range(0, len(distances), n)]

for dist in chunks:
  train_predict_and_save_model(train_df=prepared_df, features=features_2, training_sample=training_sample, testing_sample=testing_sample, pred_df=predictions_df, model='model2', params=params, dist=dist)

train_predict_and_save_model(train_df=prepared_df, features=features_2, training_sample=training_sample, testing_sample=testing_sample, pred_df=predictions_df,  model='model2', params=params)

# COMMAND ----------

#incremental training

# params=  {
#   'colsample_bytree': 0.6,
#   'max_depth': 9,
#   'min_child_weight': 5,
#   'eval_metric': 'mae',
#   'subsample': 0.6,
#   'updater':'refresh',
#   'process_type': 'update',
#   'refresh_leaf': True,
#   'silent': False,
# #    'colsample': 1.0,
#    'eta': 0.05}

# per farm
for farm in range(1,7):
  
  train_window_start, test_window_start = initialization()

  while test_window_start<=END_TEST_PERIOD:
    
    test_window_end = test_window_start+dt.timedelta(hours=47)
    train_window_end = train_window_start+dt.timedelta(hours=35)
    
    training_sample = train_sample(prepared_df, train_window_start, train_window_end)
    testing_sample = write_sample(prepared_df, train_window_start, test_window_end)
#     trained_model='model2_farm_' + str(farm)
    train_predict_and_save_model(train_df=prepared_df, 
                                 features=features_2, 
                                 training_sample=training_sample, 
                                 testing_sample=testing_sample, 
                                 pred_df=predictions_df,  
                                 model='model2', 
                                 params=params, farm=farm
#                                  , trained_model=trained_model
                                )

    test_window_start=test_window_end+dt.timedelta(hours=37)
    train_window_start=train_window_end+dt.timedelta(hours=49)

# per dist
for dist in chunks:
  
  train_window_start, test_window_start = initialization()

  while test_window_start<=END_TEST_PERIOD:
    
    test_window_end=test_window_start+dt.timedelta(hours=47)
    train_window_end=train_window_start+dt.timedelta(hours=35)
    
    training_sample = train_sample(prepared_df, train_window_start, train_window_end)
    testing_sample = write_sample(prepared_df, train_window_start, test_window_end)
#     trained_model='model2_dist_' + str(dist)
    train_predict_and_save_model(train_df=prepared_df,
                                 features=features_2,
                                 training_sample=training_sample,
                                 testing_sample=testing_sample,
                                 pred_df=predictions_df, 
                                 model='model2', params=params, 
                                 dist=dist
#                                  , trained_model= trained_model
                                )

    test_window_start=test_window_end+dt.timedelta(hours=37)
    train_window_start=train_window_end+dt.timedelta(hours=49)

train_window_start, test_window_start = initialization()

# overall
while test_window_start<=END_TEST_PERIOD:
  
  test_window_end=test_window_start+dt.timedelta(hours=47)
  train_window_end=train_window_start+dt.timedelta(hours=35)
  training_sample = train_sample(prepared_df, train_window_start, train_window_end)
  testing_sample = write_sample(prepared_df, train_window_start, test_window_end)
#   trained_model='model2_all'
  train_predict_and_save_model(train_df=prepared_df,
                               features=features_2, 
                               training_sample=training_sample, 
                               testing_sample=testing_sample, 
                               pred_df=predictions_df, 
                               model='model2', 
                               params=params 
#                                trained_model=trained_model
                              )

  test_window_start=test_window_end+dt.timedelta(hours=37)
  train_window_start=train_window_end+dt.timedelta(hours=49)

# COMMAND ----------

predictions_df

# COMMAND ----------

# Ensemble it all
lm=LinearRegression()
X_train=predictions_df.set_index('date').loc[:'2010'].reset_index().drop('date',axis=1)
y_train=prepared_df.set_index('date').loc[:'2010','wp']
lm.fit(X_train,y_train)
X_pred=predictions_df.drop('date',axis=1)
predictions_df['ensemble_predictions']=lm.predict(X_pred)

# COMMAND ----------

predictions=predictions_df.groupby(['date','farm'])['ensemble_predictions'].agg('mean').reset_index()

# COMMAND ----------

dataframes=dict()
df_list_str=['wp1','wp2','wp3','wp4','wp5','wp6']
for i,wp in enumerate(df_list_str):
  dataframes[wp]=pd.DataFrame()
  dataframes[wp]=predictions[predictions['farm']==(i+1)].drop('farm',axis=1).rename(columns={'ensemble_predictions':'wp'+str(i+1)}).reset_index(drop=True)

# COMMAND ----------

from functools import reduce
df_list=dataframes.values()
predictions=reduce(lambda x, y: pd.merge(x, y, on = 'date'), df_list) #https://stackoverflow.com/questions/38089010/merge-a-list-of-pandas-dataframes

# COMMAND ----------

predictions['date']=predictions['date'].apply(lambda x: x.strftime('%Y%m%d%H'))

# COMMAND ----------

mask=data['wp1'].isna()
test_dates=(data[mask]['date']).astype(str)

# COMMAND ----------

submission=predictions[predictions['date'].isin(test_dates)]

# COMMAND ----------

submission

# COMMAND ----------

submission.to_csv('/dbfs/FileStore/ChallengeWindPower/Phase2/predictions.csv', index=False, sep=';')

# COMMAND ----------

#Comment faire pour avoir les bons index pour le retraining?
# Quand on va retrain on va retrain sur les données prédites?
# Bien reréfléchir à ce qu'on peut utiliser à date ou pas--> a priori on entraîne jusqu'à aujoourd'hui mais on peut transformer les données jusqu'au bout du data set

# COMMAND ----------

# Windows clustering per farm

# from sklearn.cluster import KMeans
# import numpy as np

# clusters_df=pd.DataFrame()

# for farm in range(1,7):
#   df=(forecasts_and_wp_df[forecasts_and_wp_df.farm==farm]
#       .groupby('forecast_date').agg('mean')
#       .sort_values('forecast_date'))
  
#   df_split=np.split(df['ws.angle'],len(df)/12) #checked that it is always the same size
  
# #   We can't use future data for training:
#   df_split_train=[window for window in df_split if window.index[11].year<=2010]
  
#   kmeans=KMeans(n_clusters=6)
#   kmeans.fit(df_split_train)
#   clusters=kmeans.predict(df_split)
  
#   df['farm_cluster']=0
#   df['begin']=0
#   for i, cluster in enumerate(clusters):
#     df['farm_cluster'].iloc[i*12:12*(i+1)]=cluster
#     df['begin'].iloc[i*12:12*(i+1)]=df['hour'].iloc[i*12]
#   df['begin']=df.begin.astype(int)
#   clusters_df=pd.concat([clusters_df,df])
  
#   21/07 16h55 pas sûr que Leustagos ait fait la même chose il semblerait qu'il conserve la feature "distance"