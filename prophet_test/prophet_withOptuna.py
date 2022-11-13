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

pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)

search_count = 100  #探索回数

# 目的関数の設定(エアコンの出荷台数用)
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
    m.add_regressor('y_buildingMaker')
    df_buf = pd.concat([df_train, df_buildingMaker.loc[:47, ['y_buildingMaker']]], axis=1)
    m.fit(df_buf)
    df_future = m.make_future_dataframe(periods=test_length,freq='MS')
    df_future = pd.concat([df_future, df_buildingMaker.loc[:, ['y_buildingMaker']]], axis=1)
    # print(df_future)
    df_pred = m.predict(df_future) 
    preds = df_pred.tail(len(df_test))
    
    val_rmse = np.sqrt(mean_squared_error(df_test.y, preds.yhat))
    return val_rmse

# 目的関数の設定(外部データ用：ビルの建設開始，製造業)
def objective_buoldingMaker(trial):
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
    m.fit(df_train_buildingMaker)
    df_future = m.make_future_dataframe(periods=test_length_buildingMaker,freq='MS')
    df_pred = m.predict(df_future) 
    preds = df_pred.tail(len(df_test_buildingMaker))
    
    val_rmse = np.sqrt(mean_squared_error(df_test_buildingMaker.y, preds.yhat))
    return val_rmse


####################################################################################
# 予測のもとになる外部データの読み込み
datapath_buildingMaker = "./学習用データ/remake/ビル_BCS会社エリア製造業の建物.csv"
df_buildingMaker = pd.read_csv(datapath_buildingMaker)
df_buildingMaker.columns = ['ds', 'y', 'Category', 'Item', 'SubItem']  #Date:ds, Index:y

test_length_buildingMaker = 12
df_train_buildingMaker = df_buildingMaker.iloc[:-test_length_buildingMaker]
df_test_buildingMaker = df_buildingMaker.iloc[-test_length_buildingMaker:]

# ハイパーパラメータの探索の実施
study = optuna.create_study(direction="minimize")
study.optimize(objective_buoldingMaker, n_trials=search_count)

# 最適パラメータでモデル学習
m = Prophet(**study.best_params)
m.fit(df_train_buildingMaker)

# 予測の実施（学習期間＋テスト期間）
df_future_buildingMaker = m.make_future_dataframe(periods=test_length_buildingMaker,
                                    freq='MS')
df_pred_buildingMaker = m.predict(df_future_buildingMaker) 

# 元のデータセットに予測値を結合
df_buildingMaker['Predict'] = df_pred_buildingMaker['yhat']
df_buildingMaker.columns = ['ds','Index','Category','Item','SubItem','y_buildingMaker']
# #予測値
# train_pred_buildingMaker = df_buildingMaker.iloc[:-test_length_buildingMaker].loc[:, 'Predict']
# test_pred_buildingMaker = df_pred_buildingMaker.iloc[-test_length_buildingMaker:].loc[:, 'Predict']
# #実測値
# y_train_buildingMaker = df_buildingMaker.iloc[:-test_length_buildingMaker].loc[:, 'y']
# y_test_buildingMaker = df_buildingMaker.iloc[-test_length_buildingMaker:].loc[:, 'y']


# df_buildingMaker = df_buildingMaker[48:]
# df_buildingMaker = df_buildingMaker.loc[:,['ds', 'y_buildingMaker']]
# print(df_buildingMaker)

###################################################################################
#予測対象のデータ読み込み
# data_path = "./AirPassengers.csv"
# data_path = "./dataset.csv"
data_path = "./学習用データ/PAC_Total_1412.csv"
df = pd.read_csv(data_path)
df.columns = ['ds', 'y','item'] #日付：DS、目的変数：y

# print(df)

# 学習データとテストデータの分割
test_length = 12
df_train = df.iloc[:-test_length]
df_test = df.iloc[-test_length:]

# # 外部変数の連結(ビル建築開始データ2010-2014予測値)
# buff = df_buildingMaker.loc[:47, ['y_buildingMaker']]
# df_train = pd.concat([df_train, buff], axis=1)
# buff = df_buildingMaker.loc[48:,['ds','y_buildingMaker']]
# df_train = pd.concat([df_train, buff], axis=0)
# print(df_train)


# ハイパーパラメータの探索の実施
study = optuna.create_study(direction="minimize")
study.optimize(objective, n_trials=search_count)

# 最適パラメータの出力
print(f"The best parameters are : \n {study.best_params}")


# 最適パラメータでモデル学習
m = Prophet(**study.best_params)
m.add_regressor('y_buildingMaker')
df_buf = pd.concat([df_train, df_buildingMaker.loc[:47, ['y_buildingMaker']]], axis=1)
m.fit(df_buf)

# 予測の実施（学習期間＋テスト期間）
df_future = m.make_future_dataframe(periods=test_length, freq='MS')
# df_future = df_train.loc[48:, ['y_buildingMaker']]
df_future = pd.concat([df_future, df_buildingMaker.loc[:, ['y_buildingMaker']]], axis=1)
print(df_future)
df_pred = m.predict(df_future) 
# 元のデータセットに予測値を結合
df['Predict'] = df_pred['yhat']
#予測値
train_pred = df.iloc[:-test_length].loc[:, 'Predict']
test_pred = df.iloc[-test_length:].loc[:, 'Predict']
#実測値
y_train = df.iloc[:-test_length].loc[:, 'y']
y_test = df.iloc[-test_length:].loc[:, 'y']

print(df)

###評価###
print('***********************')
print(mean_absolute_percentage_error(y_test, test_pred))
print('***********************')



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
plt.ylabel('amount')      #タテ軸のラベル
plt.xlabel('month')       #ヨコ軸のラベル
plt.show()



