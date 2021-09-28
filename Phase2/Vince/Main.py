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

training_end_dates = ['2010', '2011-06']
for ind, date in enumerate(training_end_dates):
  
  train_lr=ws_angle_df.set_index('date').loc[:date]
  train_lr.dropna(subset=['wp'],inplace=True)

  X_train=full_pipeline.fit_transform(train_lr)
  y_train=train_lr.wp

  lm = LinearRegression()
  lm.fit(X_train, y_train)

  ws_angle_df['ws_angle'+str(ind)]=lm.predict(full_pipeline.fit_transform(ws_angle_df))

# COMMAND ----------

window_size=3

for ind in [0,1]:
  for i in range(1,window_size+1):
    ws_angle_df['ws_angle'+str(ind)+'_p'+str(i)]=ws_angle_df.groupby(['farm', 'start'])['ws_angle'+str(ind)].shift(i)
    ws_angle_df['ws_angle'+str(ind)+'_p'+str(i)].fillna(method='bfill',inplace=True)
    ws_angle_df['ws_angle'+str(ind)+'_n'+str(i)]=ws_angle_df.groupby(['farm', 'start'])['ws_angle'+str(ind)].shift(-i)
    ws_angle_df['ws_angle'+str(ind)+'_n'+str(i)].fillna(method='ffill',inplace=True)

# ws_angle_df=ws_angle_df[['date','farm','hors','wp','ws_angle','ws','ws_angle_p1','ws_angle_p2','ws_angle_p3','ws_angle_n1','ws_angle_n2','ws_angle_n3']].rename(columns={'hors':'dist'})

# COMMAND ----------

ws_angle_df = ws_angle_df.drop(['u', 'v', 'wd', 'start', 'start_hour', 'wd_cut', 'hour'], axis=1).rename(columns={'hors':'dist'})

# COMMAND ----------

# clustering per farm

from sklearn.cluster import KMeans
import numpy as np

clusters_df=pd.DataFrame()
training_end_dates = [dt.datetime(2010,12,31), dt.datetime(2011,6,30)]

for farm in range(1,7):
  for ind, date in enumerate(training_end_dates):
    vect=ws_angle_df.loc[ws_angle_df['farm']==farm].set_index('date')['ws_angle'+str(ind)]
    X_tot=np.split(vect,len(vect)/12) #checked that it is always the same size
    X_train=[window for window in X_tot if window.index[11]<=date]
#   training and predictions
    kmeans=KMeans(n_clusters=6)
    kmeans.fit(X_train)
    clusters=kmeans.predict(X_tot)
# Filling of the dataset
    if ind==0:
      results_df=ws_angle_df[['date','farm','dist']].loc[ws_angle_df.farm==farm]
      results_df['hour']=results_df['date'].dt.hour
      results_df['window_start']=0
      results_df['window_instance_pos']=0
    results_df['farm_cluster' + str(ind)]=0
    for i, cluster in enumerate(clusters):
      results_df['window_instance_pos'].iloc[i*12:12*(i+1)]=range(1,13) #checké si on a le même entre un cluster par farm et un cluster total
      results_df['farm_cluster' + str(ind)].iloc[i*12:12*(i+1)]=cluster
      results_df['window_start'].iloc[i*12:12*(i+1)]=results_df['hour'].iloc[i*12]
  results_df['window_start']=results_df.window_start.astype(int)
#   Concatenation
  clusters_df=pd.concat([clusters_df,results_df])
  clusters_df.drop(['hour'],axis=1)

# COMMAND ----------

# overall clustering
for ind, date in enumerate(training_end_dates):
  vect=ws_angle_df.set_index('date')['ws_angle'+str(ind)]
  X_tot=np.split(vect,len(vect)/12) #1 farm per window?
#   We can't use future data for training
  X_train=[window for window in X_tot if window.index[11]<=date]
#   training and predictions
  kmeans=KMeans(n_clusters=24)
  kmeans.fit(X_train)
  clusters=kmeans.predict(X_tot)
# Filling of the dataset
  clusters_df['general_cluster' + str(ind)]=0
  for i, cluster in enumerate(clusters):
    clusters_df['general_cluster'+str(ind)].iloc[i*12:12*(i+1)]=cluster

# COMMAND ----------

# Join everything

prepared_df=pd.merge(ws_angle_df,clusters_df,how='left',left_on=['date','farm','dist'],right_on=['date','farm','dist'])

prepared_df['month']=prepared_df['date'].dt.month
prepared_df['year']=prepared_df['date'].dt.year

# COMMAND ----------

def train_model_per_farm(farm,model,df,features_list,predictions_column,predictions_df,end_date_str):
  
  filtered_df=df[df['farm']==farm].set_index('date')
  filtered_df.dropna(subset=['wp'],inplace=True)
  X_train=filtered_df.loc[:end_date_str][features_list]
  y_train=filtered_df.loc[:end_date_str]['wp']
  model.fit(X_train,y_train)
  X_pred=df.loc[df['farm']==farm,features_list]  
  predictions_df.loc[df['farm']==farm,predictions_column]=model.predict(X_pred)

# COMMAND ----------

predictions_df=prepared_df[['date','farm','dist']].copy()

# COMMAND ----------

# model 1 per farm

from sklearn.ensemble import GradientBoostingRegressor

features_list0=['ws','farm','dist','ws_angle0','ws_angle0_p1', 'ws_angle0_n1', 'ws_angle0_p2', 'ws_angle0_n2',
       'ws_angle0_p3', 'ws_angle0_n3','hour','month','year']

features_list1=['ws','farm','dist','ws_angle1','ws_angle1_p1', 'ws_angle1_n1', 'ws_angle1_p2', 'ws_angle1_n2',
       'ws_angle1_p3', 'ws_angle1_n3','hour','month','year']


for features_list, end_date in zip([features_list0, features_list1], ["2010", "2011-06"]):
  if end_date=="2010":
    predictions_column='predictions_model1_farm0'
  else:
    predictions_column='predictions_model1_farm1'
  for farm in range(1,7):
    model=GradientBoostingRegressor(random_state=0)
    train_model_per_farm(farm,model,prepared_df,features_list,predictions_column,predictions_df, end_date)

# COMMAND ----------

# model 2 per farm

features_list0=['farm','dist','ws_angle0_p1', 'ws_angle0_p2',
       'ws_angle0_p3','hour','month','farm_cluster0','general_cluster0','window_start','window_instance_pos']

features_list1=['farm','dist','ws_angle1_p1', 'ws_angle1_p2',
       'ws_angle1_p3','hour','month','farm_cluster1','general_cluster1','window_start','window_instance_pos']


for features_list, end_date in zip([features_list0, features_list1], ["2010", "2011-06"]):
  if end_date=="2010":
    predictions_column='predictions_model2_farm0'
  else:
    predictions_column='predictions_model2_farm1'
    
  for farm in range(1,7):
    model=GradientBoostingRegressor(random_state=0)
    train_model_per_farm(farm,model,prepared_df,features_list,predictions_column,predictions_df, end_date)

# COMMAND ----------

n = 3 # chunk length
distances=range(1,49)
chunks = [distances[i:i+n] for i in range(0, len(distances), n)]

# COMMAND ----------

def train_model_per_dist(dist:range,
                         model,
                         df:pd.DataFrame,
                         features_list:list,
                         predictions_column:str,
                         predictions_df:pd.DataFrame,
                         end_date:str
                        ):
  
  mask=df['dist'].isin(dist)
  filtered_df=df.loc[mask].set_index('date')
  filtered_df.dropna(subset=['wp'],inplace=True)
  X_train=filtered_df.loc[:end_date,features_list] 
  y_train=filtered_df.loc[:end_date,'wp']
  model.fit(X_train,y_train)
  X_pred=df.loc[mask,features_list]  
  predictions_df.loc[mask,predictions_column]=model.predict(X_pred)

# COMMAND ----------

# model 1 per distance

features_list0=['ws','farm','dist','ws_angle0','ws_angle0_p1', 'ws_angle0_n1', 'ws_angle0_p2', 'ws_angle0_n2',
       'ws_angle0_p3', 'ws_angle0_n3','hour','month','year']

features_list1=['ws','farm','dist','ws_angle1','ws_angle1_p1', 'ws_angle1_n1', 'ws_angle1_p2', 'ws_angle1_n2',
       'ws_angle1_p3', 'ws_angle1_n3','hour','month','year']

predictions_column='predictions_model1_dist'

for features_list, end_date in zip([features_list0, features_list1], ["2010", "2011-06"]):
  if end_date=="2010":
    predictions_column='predictions_model1_dist0'
  else:
    predictions_column='predictions_model1_dist1'
  for dist in chunks:
    model=GradientBoostingRegressor(random_state=0)
    train_model_per_dist(dist,model,prepared_df,features_list,predictions_column,predictions_df, end_date)

# COMMAND ----------

# model 2 per distance

features_list0=['farm','dist','ws_angle0_p1', 'ws_angle0_p2',
       'ws_angle0_p3','hour','month','farm_cluster0','general_cluster0','window_start','window_instance_pos']
features_list1=['farm','dist','ws_angle1_p1', 'ws_angle1_p2',
       'ws_angle1_p3','hour','month','farm_cluster1','general_cluster1','window_start','window_instance_pos']

for features_list, end_date in zip([features_list0, features_list1], ["2010", "2011-06"]):
  if end_date=="2010":
    predictions_column='predictions_model2_dist0'
  else:
    predictions_column='predictions_model2_dist1'
  for dist in chunks:
    model=GradientBoostingRegressor(random_state=0)
    train_model_per_dist(dist,model,prepared_df,features_list,predictions_column,predictions_df, end_date)

# COMMAND ----------

#Overall models

# model 1
features_list0=['ws','farm','dist','ws_angle0','ws_angle0_p1', 'ws_angle0_n1', 'ws_angle0_p2', 'ws_angle0_n2',
       'ws_angle0_p3', 'ws_angle0_n3','hour','month','year']

features_list1=['ws','farm','dist','ws_angle1','ws_angle1_p1', 'ws_angle1_n1', 'ws_angle1_p2', 'ws_angle1_n2',
       'ws_angle1_p3', 'ws_angle1_n3','hour','month','year']


for features_list, end_date in zip([features_list0, features_list1], ["2010", "2011-06"]):
  if end_date=="2010":
    predictions_column='predictions_model1_all0'
  else:
    predictions_column='predictions_model1_all1'

  model=GradientBoostingRegressor(random_state=0)

  X_train=prepared_df.dropna(subset=['wp']).set_index('date').loc[:end_date,features_list] 
  y_train=prepared_df.dropna(subset=['wp']).set_index('date').loc[:end_date,'wp']
  model.fit(X_train,y_train)

  X_pred=prepared_df.loc[:,features_list]  
  predictions_df.loc[:,predictions_column]=model.predict(X_pred)

# COMMAND ----------

# model 2
features_list0=['farm','dist','ws_angle0_p1', 'ws_angle0_p2',
       'ws_angle0_p3','hour','month','farm_cluster0','general_cluster0','window_start','window_instance_pos']
features_list1=['farm','dist','ws_angle1_p1', 'ws_angle1_p2',
       'ws_angle1_p3','hour','month','farm_cluster1','general_cluster1','window_start','window_instance_pos']


for features_list, end_date in zip([features_list0, features_list1], ["2010", "2011-06"]):
  if end_date=="2010":
    predictions_column='predictions_model2_all0'
  else:
    predictions_column='predictions_model2_all1'

  model=GradientBoostingRegressor(random_state=0)

  X_train=prepared_df.dropna(subset=['wp']).set_index('date').loc[:end_date,features_list] 
  y_train=prepared_df.dropna(subset=['wp']).set_index('date').loc[:end_date,'wp']
  model.fit(X_train,y_train)

  X_pred=prepared_df.loc[:,features_list]  
  predictions_df.loc[:,predictions_column]=model.predict(X_pred)

# COMMAND ----------

# Ensemble it all
features_list0=['predictions_model1_farm0', 'predictions_model2_farm0', 'predictions_model1_dist0', 'predictions_model2_dist0', 'predictions_model1_all0', 'predictions_model2_all0' ]
features_list1=['predictions_model1_farm1', 'predictions_model2_farm1', 'predictions_model1_dist1', 'predictions_model2_dist1', 'predictions_model1_all1', 'predictions_model2_all1' ]

predictions_df['wp']=prepared_df['wp']

for features_list, end_date in zip([features_list0, features_list1], ["2010", "2011-06"]):
  if end_date=="2010":
    ensemble_column='ensemble_predictions0'
  else:
    ensemble_column='ensemble_predictions1'
  lm=LinearRegression()
  X_train=predictions_df.dropna(subset=['wp']).set_index('date').loc[:end_date, features_list].reset_index().drop('date',axis=1)
  y_train=prepared_df.dropna(subset=['wp']).set_index('date').loc[:end_date,'wp']
  lm.fit(X_train,y_train)
  X_pred=predictions_df.loc[:, features_list]
  predictions_df[ensemble_column]=lm.predict(X_pred)

# COMMAND ----------

pred1 = predictions_df.set_index('date')['ensemble_predictions0'].loc[:'2011-06'].reset_index(drop=True)
pred2 = predictions_df.set_index('date')['ensemble_predictions1'].loc['2011-07':].reset_index(drop=True)

pred=pd.concat([pred1,pred2], axis=0, ignore_index=True)

# COMMAND ----------

predictions_df['ensemble_predictions']=pred

# COMMAND ----------

predictions_df.loc[predictions_df.date=="2011-01-01 1:00:00"]

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