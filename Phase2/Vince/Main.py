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

import datetime as dt

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

train_pivot

# COMMAND ----------

wind_forecasts_df

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

train_lr=ws_angle_df.set_index('date').loc[:'2010']

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
  X_train=[window for window in X_tot if window.index[11].year<=2010]
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
X_tot=np.split(vect,len(vect)/12) #1 farm per window?
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

clusters_df

# COMMAND ----------

# Join everything

prepared_df=pd.merge(ws_angle_df,clusters_df,how='left',left_on=['date','farm','dist'],right_on=['date','farm','dist'])

prepared_df['month']=prepared_df['date'].dt.month
prepared_df['year']=prepared_df['date'].dt.year

# COMMAND ----------



# COMMAND ----------

def train_model_per_farm(farm, training_year, testing_year, df,features_list,predictions_column,predictions_df, model_version:str, trained_model=None):
  
  params=  {'colsample_bytree': 0.6,
   'max_depth': 9,
   'min_child_weight': 5,
   'eval_metric': 'mae',
   'subsample': 0.6,
   'colsample': 1.0,
   'eta': 0.05}
  
  filtered_df=df[df['farm']==farm].set_index('date')
  X_train=filtered_df.dropna().loc[str(training_year-1):str(training_year),features_list]
  y_train=filtered_df.dropna().loc[str(training_year-1):str(training_year),'wp']
  xg_train = xgb.DMatrix(X_train, label=y_train)
  if trained_model is None:
    model=xgb.train(params=params, dtrain=xg_train)
    print('new model')
  else:
    model=xgb.train(params=params, dtrain=xg_train, xgb_model=trained_model)
    print('use previous model')
  
  model.save_model(model_version+'_farm'+str(farm))
  X_pred=filtered_df.loc[str(testing_year):,features_list]
  xg_pred=xgb.DMatrix(X_pred)
  mask=(predictions_df.date.dt.year>=testing_year) & (predictions_df['farm']==farm)
  predictions_df.loc[mask,predictions_column]=model.predict(xg_pred)

# COMMAND ----------

predictions_df=prepared_df[['date','farm','dist']].copy()

# COMMAND ----------

# model 1 per farm

import xgboost as xgb

features_list=['ws','farm','dist','ws_angle','ws_angle_p1', 'ws_angle_n1', 'ws_angle_p2', 'ws_angle_n2',
       'ws_angle_p3', 'ws_angle_n3','hour','month','year']
predictions_column='predictions_model1_farm'

for farm in range(1,7):
  train_model_per_farm(farm, 2010, 2009, prepared_df, features_list,predictions_column,predictions_df, model_version='model1')

# COMMAND ----------

# model 2 per farm

features_list=['farm','dist','ws_angle_p1', 'ws_angle_p2',
       'ws_angle_p3','hour','month','farm_cluster','general_cluster','window_start','window_instance_pos']

predictions_column='predictions_model2_farm'

for farm in range(1,7):
  train_model_per_farm(farm, 2010, 2009, prepared_df, features_list,predictions_column,predictions_df, model_version='model2')

# COMMAND ----------

n = 3 # chunk length
distances=range(1,49)
chunks = [distances[i:i+n] for i in range(0, len(distances), n)]

# COMMAND ----------

def train_model_per_dist(dist:range,
                         df:pd.DataFrame,
                         training_year,
                         testing_year,
                         features_list:list,
                         predictions_column:str,
                         predictions_df:pd.DataFrame,
                         model_version,
                        trained_model=None):
  
  params=  {'colsample_bytree': 0.6,
 'max_depth': 9,
 'min_child_weight': 5,
 'eval_metric': 'mae',
 'subsample': 0.6,
 'colsample': 1.0,
 'eta': 0.05}
  
  mask=(df['dist'].isin(dist))
  filtered_df=df.loc[mask].set_index('date')
  X_train=filtered_df.dropna().loc[str(training_year-1):str(training_year),features_list] 
  y_train=filtered_df.dropna().loc[str(training_year-1):str(training_year),'wp']
  
  xg_train = xgb.DMatrix(X_train, label=y_train)
  if trained_model is None:
    model=xgb.train(params=params, dtrain=xg_train)
    print('new model')
  else:
    model=xgb.train(params=params, dtrain=xg_train, xgb_model=trained_model)
    
  model.save_model(model_version+'_dist'+str(dist))
  X_pred=filtered_df.loc[str(testing_year):,features_list]
  xg_pred=xgb.DMatrix(X_pred)
  mask_2=(predictions_df.date.dt.year>=testing_year) & mask
  predictions_df.loc[mask_2,predictions_column]=model.predict(xg_pred)

# COMMAND ----------

# model 1 per distance

features_list=['ws','farm','dist','ws_angle','ws_angle_p1', 'ws_angle_n1', 'ws_angle_p2', 'ws_angle_n2',
       'ws_angle_p3', 'ws_angle_n3','hour','month','year']
predictions_column='predictions_model1_dist'

for dist in chunks:
  train_model_per_dist(dist, prepared_df, 2010, 2009, features_list,predictions_column,predictions_df, model_version='model1')

# COMMAND ----------

# model 2 per distance

features_list=['farm','dist','ws_angle_p1', 'ws_angle_p2',
       'ws_angle_p3','hour','month','farm_cluster','general_cluster','window_start','window_instance_pos']
predictions_column='predictions_model2_dist'

for dist in chunks:
  train_model_per_dist(dist,prepared_df, 2010, 2009, features_list,predictions_column,predictions_df, model_version='model2')

# COMMAND ----------

#Overall models

training_year = 2010
testing_year = 2009

# model 1

features_list=['ws','farm','dist','ws_angle','ws_angle_p1', 'ws_angle_n1', 'ws_angle_p2', 'ws_angle_n2',
       'ws_angle_p3', 'ws_angle_n3','hour','month','year']

predictions_column='predictions_model1_all'

X_train=prepared_df.set_index('date').loc[str(training_year-1):str(training_year),features_list] 
y_train=prepared_df.set_index('date').loc[str(training_year-1):str(training_year),'wp']
xg_train = xgb.DMatrix(X_train, label=y_train)
model=xgb.train(params=params, dtrain=xg_train)
model.save_model('model1_all')
X_pred=prepared_df.loc[prepared_df.date.dt.year>=testing_year,features_list]
xg_pred=xgb.DMatrix(X_pred)
predictions_df.loc[prepared_df.date.dt.year>=testing_year,predictions_column]=model.predict(xg_pred)

# model 2

features_list=['farm','dist','ws_angle_p1', 'ws_angle_p2',
       'ws_angle_p3','hour','month','farm_cluster','general_cluster','window_start','window_instance_pos']

predictions_column='predictions_model2_all'

X_train=prepared_df.set_index('date').loc[str(training_year-1):str(training_year),features_list] 
y_train=prepared_df.set_index('date').loc[str(training_year-1):str(training_year),'wp']
xg_train = xgb.DMatrix(X_train, label=y_train)
model=xgb.train(params=params, dtrain=xg_train)
model.save_model('model2_all')
X_pred=prepared_df.loc[prepared_df.date.dt.year>=testing_year,features_list]
xg_pred=xgb.DMatrix(X_pred)
predictions_df.loc[prepared_df.date.dt.year>=testing_year,predictions_column]=model.predict(xg_pred)

# COMMAND ----------

# model 1 per farm

import xgboost as xgb

features_list=['ws','farm','dist','ws_angle','ws_angle_p1', 'ws_angle_n1', 'ws_angle_p2', 'ws_angle_n2',
       'ws_angle_p3', 'ws_angle_n3','hour','month','year']
predictions_column='predictions_model1_farm'

for farm in range(1,7):
  trained_model='model1_farm'+str(farm)
  train_model_per_farm(farm, 2011, 2011, prepared_df, features_list,predictions_column,predictions_df, model_version='model1', trained_model = trained_model)

# COMMAND ----------

# model 2 per farm

features_list=['farm','dist','ws_angle_p1', 'ws_angle_p2',
       'ws_angle_p3','hour','month','farm_cluster','general_cluster','window_start','window_instance_pos']

predictions_column='predictions_model2_farm'

for farm in range(1,7):
  trained_model='model2_farm'+str(farm)
  train_model_per_farm(farm, 2011, 2011, prepared_df, features_list,predictions_column,predictions_df, model_version='model2', trained_model=trained_model)

# COMMAND ----------

# model 1 per distance

features_list=['ws','farm','dist','ws_angle','ws_angle_p1', 'ws_angle_n1', 'ws_angle_p2', 'ws_angle_n2',
       'ws_angle_p3', 'ws_angle_n3','hour','month','year']
predictions_column='predictions_model1_dist'

for dist in chunks:
  trained_model='model1_dist'+str(dist)
  train_model_per_dist(dist, prepared_df, 2011, 2011, features_list,predictions_column,predictions_df, model_version='model1', trained_model=trained_model)

# COMMAND ----------

# model 2 per distance

features_list=['farm','dist','ws_angle_p1', 'ws_angle_p2',
       'ws_angle_p3','hour','month','farm_cluster','general_cluster','window_start','window_instance_pos']
predictions_column='predictions_model2_dist'

for dist in chunks:
  trained_model='model2_dist'+str(dist)
  train_model_per_dist(dist,prepared_df, 2011, 2011, features_list,predictions_column,predictions_df, model_version= 'model2', trained_model=trained_model)

# COMMAND ----------

#Overall models

training_year = 2011
testing_year = 2011

# model 1
features_list=['ws','farm','dist','ws_angle','ws_angle_p1', 'ws_angle_n1', 'ws_angle_p2', 'ws_angle_n2',
       'ws_angle_p3', 'ws_angle_n3','hour','month','year']

predictions_column='predictions_model1_all'

X_train=prepared_df.set_index('date').loc[str(training_year-1):str(training_year),features_list] 
y_train=prepared_df.set_index('date').loc[str(training_year-1):str(training_year),'wp']
xg_train = xgb.DMatrix(X_train, label=y_train)
model=xgb.train(params=params, dtrain=xg_train, xgb_model='model1_all')

X_pred=prepared_df.loc[prepared_df.date.dt.year>=testing_year,features_list]
xg_pred=xgb.DMatrix(X_pred)
predictions_df.loc[prepared_df.date.dt.year>=testing_year,predictions_column]=model.predict(xg_pred)

# model 2

features_list=['farm','dist','ws_angle_p1', 'ws_angle_p2',
       'ws_angle_p3','hour','month','farm_cluster','general_cluster','window_start','window_instance_pos']

predictions_column='predictions_model2_all'

X_train=prepared_df.set_index('date').loc[str(training_year-1):str(training_year),features_list] 
y_train=prepared_df.set_index('date').loc[str(training_year-1):str(training_year),'wp']
xg_train = xgb.DMatrix(X_train, label=y_train)
model=xgb.train(params=params, dtrain=xg_train, xgb_model='model2_all')

X_pred=prepared_df.loc[prepared_df.date.dt.year>=testing_year,features_list]
xg_pred=xgb.DMatrix(X_pred)
predictions_df.loc[prepared_df.date.dt.year>=testing_year,predictions_column]=model.predict(xg_pred)

# COMMAND ----------

# Ensemble it all
lm=LinearRegression()
X_train=predictions_df.set_index('date').loc[:'2010'].reset_index().drop('date',axis=1)
y_train=prepared_df.set_index('date').loc[:'2010','wp']
lm.fit(X_train,y_train)
X_pred=predictions_df.drop('date',axis=1)
predictions_df['ensemble_predictions']=lm.predict(X_pred)

# COMMAND ----------

predictions_df[predictions_df.isna().any(axis=1)]

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