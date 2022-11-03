# ライブラリーの読み込み
import numpy as np
import pandas as pd
import optuna
from prophet import Prophet
from prophet.diagnostics import cross_validation
from prophet.diagnostics import performance_metrics
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_percentage_error
import matplotlib.pyplot as plt

# 目的関数の設定
def objective(trial):
    params = {'changepoint_prior_scale' : 
                 trial.suggest_uniform('changepoint_prior_scale',
                                       0.001,0.5
                                      ),
              'seasonality_prior_scale' : 
                 trial.suggest_uniform('seasonality_prior_scale',
                                       0.01,10
                                      ),
              'seasonality_mode' : 
                 trial.suggest_categorical('seasonality_mode',
                                           ['additive', 'multiplicative']
                                          ),
              'changepoint_range' : 
                  trial.suggest_discrete_uniform('changepoint_range', 
                                                 0.8, 0.95, 
                                                 0.001),
              'n_changepoints' : 
                  trial.suggest_int('n_changepoints', 
                                    20, 35),
             }
    m = Prophet(**params)
    m.fit(df_train)
    df_future = m.make_future_dataframe(periods=test_length,freq='M')
    df_pred = m.predict(df_future) 
    preds = df_pred.tail(len(df_test))
    
    val_rmse = np.sqrt(mean_squared_error(df_test.y, preds.yhat))
    return val_rmse


#データ読み込み
data_path = "./AirPassengers.csv"
df = pd.read_csv(data_path)
df.columns = ['ds', 'y'] #日付：DS、目的変数：y

print(df.head())

# 学習データとテストデータの分割
test_length = 12
df_train = df.iloc[:-test_length]
df_test = df.iloc[-test_length:]

# ハイパーパラメータの探索の実施
study = optuna.create_study(direction="minimize")
study.optimize(objective, n_trials=100)

# 最適パラメータの出力
print(f"The best parameters are : \n {study.best_params}")


# 最適パラメータでモデル学習
m = Prophet(**study.best_params)
m.fit(df_train)

# 予測の実施（学習期間＋テスト期間）
df_future = m.make_future_dataframe(periods=test_length,
                                    freq='M')
df_pred = m.predict(df_future) 
# 元のデータセットに予測値を結合
df['Predict'] = df_pred['yhat']
#予測値
train_pred = df.iloc[:-test_length].loc[:, 'Predict']
test_pred = df.iloc[-test_length:].loc[:, 'Predict']
#実測値
y_train = df.iloc[:-test_length].loc[:, 'y']
y_test = df.iloc[-test_length:].loc[:, 'y']

# グラフ化
# fig, ax = plt.subplots()
# ax.plot(df_train.ds, df_train.y, label="actual(train dataset)")
# ax.plot(df_test.ds, df_test.y, label="actual(test dataset)")
fig, ax = plt.subplots()
ax.plot(df_train.ds, df_train.y, label="actual(train dataset)")
ax.plot(df_test.ds, df_test.y, label="actual(test dataset)")
ax.plot(df_train.ds, train_pred, linestyle="dotted", lw=2,color="m")
ax.plot(df_test.ds, test_pred, label="Prophet", linestyle="dotted", lw=2, color="m") 
plt.legend()
# df.plot(kind='line',x='ds', y='y')
plt.title('Passengers')                            #グラフタイトル
plt.ylabel('Monthly Number of Airline Passengers') #タテ軸のラベル
plt.xlabel('Month')                                #ヨコ軸のラベル
plt.show()



