#material backorder prediction
#Importing libraries

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import SGDClassifier
from sklearn.tree import DecisionTreeClassifier
from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from sklearn.calibration import CalibratedClassifierCV

from sklearn.metrics import roc_auc_score

from time import time
import joblib
import streamlit as st

import pickle


def predict(national_inv,
            lead_time,
            in_transit_qty,
            forecast_3_month,
            sales_3_month,
            min_bank,
            pieces_past_due,
            perf_6_month_avg,
            perf_12_month_avg,
            deck_risk,
            stop_auto_buy):
  '''This function takes a datapoint as input , preprocess and predict using a pretrained stacked model and returns the prediction as output.'''
  
  start=time()
  
  #preprocessing
  
  features=dict()
  features['national_inv']=national_inv
  features['lead_time']=lead_time
  features['in_transit_qty']=in_transit_qty
  features['forecast_3_month']=forecast_3_month
  features['sales_3_month']=sales_3_month
  features['min_bank']=min_bank
  features['pieces_past_due']=pieces_past_due
  features['perf_6_month_avg']=perf_6_month_avg
  features['perf_12_month_avg']=perf_12_month_avg
  features['deck_risk']=deck_risk
  features['stop_auto_buy']=stop_auto_buy
  
  # values for missing value imputation
  impute={'deck_risk': 0,
          'forecast_3_month': 0.0,
          'in_transit_qty': 0.0,
          'lead_time': 8.0,
          'min_bank': 0.0,
          'national_inv': 15.0,
          'perf_12_month_avg': 0.83,
          'perf_6_month_avg': 0.722,
          'pieces_past_due': 0.0,
          'sales_3_month': 0.48,
          'stop_auto_buy': 1}
  # columns to tranform
  skewed=['in_transit_qty','forecast_3_month','sales_3_month',
          'min_bank','pieces_past_due','reorder_point','usable_stock']
  
  skewed2=['forecast_3_month','sales_3_month','min_bank','perf_6_month_avg']
  
  to_scale=['national_inv', 'lead_time', 'in_transit_qty',
      'forecast_3_month', 'sales_3_month', 
      'min_bank', 'pieces_past_due']
  #scaler
  scaler=joblib.load('scaler.pkl')
  base_models=[]
  base_predictions=[]
  threshold=0.00815
  
  for i in range(1,16):
    base= joblib.load('model_'+str(i)+'.pkl')
    base_models.append(base)
  meta=joblib.load('metaclf_.pkl')
  
  
  #missing value imputation
  if features['perf_6_month_avg']<=-2:
    features['perf_6_month_avg']=impute['perf_6_month_avg']
  if features['perf_12_month_avg']<=-2:
    features['perf_12_month_avg']=impute['perf_12_month_avg']
  for i in features.keys():
    if ((features[i]=='NaN')|(features[i]=='nan')):
      features[i]=impute[i]
  
  #feature engg
  features['reorder_point']=np.round(((features['sales_3_month']/30)*features['lead_time'])+features['min_bank'],5)
  features['usable_stock']=np.round(features['national_inv']-features['reorder_point'],5)
  features['neg_stock']=(features['usable_stock']<0).astype('int32')
  features['zero_stock']=(features['usable_stock']==0).astype('int32')
  features['min_stock']=(features['usable_stock']<features['min_bank']).astype('int32')
  
  
  #feature transformations
  for feat in skewed:
    features[feat]= np.round(np.log(abs(features[feat])+1)*np.sign(features[feat]),5)
  for feat in skewed2:
    features[feat]=np.round((features[feat])**2,5)
  
  features['pieces_past_due']=np.round((features['pieces_past_due'])**4,5)
  features['usable_stock']=np.round(((features['usable_stock'])**2)*np.sign(features['usable_stock']),5)
  
  #scaling
  scaled=scaler.transform(np.array([features[v] for v in to_scale]).reshape(1,-1))
  for i,v in enumerate(to_scale):
    features[v]=np.round(scaled[0][i],6)
  
  all_cols=['national_inv', 'lead_time', 'in_transit_qty',
       'forecast_3_month', 'sales_3_month', 'min_bank', 'pieces_past_due',
       'perf_6_month_avg', 'perf_12_month_avg', 'deck_risk', 'stop_auto_buy',
        'reorder_point', 'usable_stock', 'neg_stock','zero_stock', 'min_stock']

  #preprocessd data
  preprocessed=np.array([features[value] for value in all_cols]).reshape(1,-1)
  
  #prediction by base models
  for model in base_models:
    base_predictions.append(model.predict_proba(preprocessed)[0][1])
  
  #meta model prediction
  meta=joblib.load('metaclf_.pkl')
  prediction=meta.predict_proba(np.array(base_predictions).reshape(1,-1))[0][1]
  prediction=(prediction>=threshold).astype('int32')
  print('prediction:',prediction )
  print('time taken: %0.2f seconds'%(time()-start))
  
  return prediction



def main():

    '''this is the main function in which we define our webpage '''
    
    st.title("Material Backorder Prediction")
    title_alignment="""
          <style>
          #material-backorder-prediction {
                 text-align: center;
                 
                      }
          </style>
          """
    st.markdown(title_alignment, unsafe_allow_html=True)
    st.markdown("Please enter the details :")

      
    # the following lines create input fields in which the user can enter 
    # the data required to make the prediction
    
    national_inv=st.number_input('National Invariance')
    lead_time=st.number_input('Lead Time')
    in_transit_qty=st.number_input('Quantity in Transit')
    forecast_3_month=st.number_input('3 months Forecast')
    sales_3_month=st.number_input('Past 3 months Sales')
    min_bank=st.number_input('Minimum Stock Recommended')
    pieces_past_due=st.number_input('Parts Overdue')
    perf_6_month_avg=st.number_input('6 months Average Perfomance')
    perf_12_month_avg=st.number_input('12 months Average Perfomance')
    deck_risk=st.radio('deck_risk', ("Yes","No"))
    stop_auto_buy=st.radio('stop_auto_buy',("Yes","No"))


    if deck_risk=="Yes":
       deck_risk=1
    else:
      deck_risk=0

    if stop_auto_buy=="Yes":
      stop_auto_buy=1
    else:
      stop_auto_buy=0
    
    result =""

    # the below line ensures that when the button called 'Predict' is clicked, 
    # the prediction function defined above is called to make the prediction 
    # and store it in the variable result
    if st.button("Predict"):
        result = predict(national_inv,
                         lead_time,
                         in_transit_qty,
                         forecast_3_month,
                         sales_3_month,
                         min_bank,
                         pieces_past_due,
                         perf_6_month_avg,
                         perf_12_month_avg,
                         deck_risk,
                         stop_auto_buy)
        if result==0:
          st.success('No Backorder risk..!')
        elif result==1:
          st.error('Alert: This Product is under backorder risk..!')
    if st.sidebar.button("Show sample data"):
       df = pd.read_csv("sample_df.csv")
       st.sidebar.dataframe(df)

   

if __name__=='__main__':
  main()
