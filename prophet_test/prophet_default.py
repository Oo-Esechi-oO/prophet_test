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

#データ読み込み
data_path = "./AirPassengers.csv"
df = pd.read_csv(data_path)
df.columns = ['ds', 'y'] #日付：DS、目的変数：y

print(df.head())

# 学習データとテストデータの分割
test_length = 12
df_train = df.iloc[:-test_length]
df_test = df.iloc[-test_length:]

# 予測モデル構築
m = Prophet()
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

