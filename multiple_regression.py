import numpy as np
from scipy.optimize import curve_fit
import pandas as pd
import csv
from scipy.stats import t
# サンプルデータの作成
y=[]
x1=[]
x2=[]
filename = 'D:/授業資料/英語/aless/Book2.csv'
with open(filename, encoding='utf8', newline='') as f:
    csvreader = csv.reader(f)
    for row in csvreader:
        try:
            x2.append(int(row[0]))
            x1.append(int(row[1]))
            y.append(int(row[2]))
        except:
            pass
data = {
    'y': y,  # 例の従属変数
    'x1': x1,      # 例の説明変数1
    'x2': x2      # 例の説明変数2
}

df = pd.DataFrame(data)

# 非線形関数の定義
def model_func(x, a, b, c):
    x1, x2 = x
    return a * np.exp(-b * x1) * np.log(x2) + c

# 説明変数と従属変数の定義
xdata = np.vstack((df['x1'], df['x2']))
ydata = df['y']

# 初期推定値の設定
initial_guess = [1.0, 1.0, 1.0]

# 非線形最小二乗法によるフィッティング
popt, pcov = curve_fit(model_func, xdata, ydata, p0=initial_guess)

# 推定された係数の出力
a, b, c = popt
print("推定された係数:")
print(f"a = {a}")
print(f"b = {b}")
print(f"c = {c}")

# 推定されたパラメータの標準誤差
perr = np.sqrt(np.diag(pcov))
print("\nパラメータの標準誤差:")
print(f"{perr[0]}")
print(f"{perr[1]}")
print(f"{perr[2]}")

# パラメータのp値を計算
alpha = 0.05
dof = max(0, len(ydata) - len(popt))  # 自由度
t_values = popt / perr  # t値の計算
p_values = [2 * (1 - t.cdf(np.abs(t_val), dof)) for t_val in t_values]
print("\nt値:")
print(f"{t_values[0]}")
print(f"{t_values[1]}")
print(f"{t_values[2]}")
print("\np値:")
print(f"{p_values[0]}")
print(f"{p_values[1]}")
print(f"{p_values[2]}")

# フィッティングの精度を確認
y_pred = model_func((df['x1'], df['x2']), a, b, c)
r_squared = 1 - np.sum((ydata - y_pred)**2) / np.sum((ydata - np.mean(ydata))**2)
print(f"\n決定係数: {r_squared}")
