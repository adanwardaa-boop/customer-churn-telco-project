import pandas as pd
from sklearn.preprocessing import StandardScaler
def engineer_features(df):
  df['AvgChargesPerMonth']= df['TotalCharges']/ (df['tenure']+1)
  categorical_cols = df.select_dtypes(include='object').columns
  df=pd.get_dummies(df, columns=categorical_cols, drop_first=True)
  x=df.drop('churn',axis=1)
  y=df['churn']
  scaler=StandardScaler()
  x_scaled=scaler.fit_transform(x)
  return x_scaled,y,x.columns

