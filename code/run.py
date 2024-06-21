# date: 2024-06-14
# hyuna
# Kalman Filter/Smooth run

from kalman_filter_smooth import Kalman
import pandas as pd
import numpy as np

# 데이터 불러오기
path = 'your path'
df = pd.read_csv(path)

# 칼만필터 파라미터 지정----------------------
# 노이즈 파라미터
R = 1  # 측정 분산
Q = np.diag([0.5])  # 과정 분산
H = np.array([[1]])  # 관측 행렬
# ------------------------------------------

# 필요 데이터
n = 600
data = df.copy()

# 칼만필터 실행-------------------------------------------------
print("kalman filter-----------------------------------")
kalman = Kalman(data[:n])
xhat, xhatminus, P, Pminus = kalman.kalman_filter()

# 예측 성능 결과값
print('예측 성능 결과값')
kalman.predict_score(data[:n], xhatminus.reshape(-1,))

# 미래 값 예측
forecast_steps = 6 # 미래 예측값
forward_n = 10 # 그 전 값이 얼마나 영향을 주는지 

A = kalman.state_matrix(data[n-forward_n:n] ,start=n-forward_n)
forecasts, forecast_covariances = kalman.forecast(xhat, P, A, forecast_steps)

# 미래값 성능 확인
print('미래 예측데이터와 실제 데이터 성능값')
kalman.predict_score(data[n:n+forecast_steps], forecasts)
print('-------------------------------------------------')
# --------------------------------------------------------------


# 칼만 스무딩 실행-------------------------------------------------
print("kalman smoothing---------------------------------")
# 칼만필터값을 넣음
xhat_smooth, P_smooth = kalman.kalman_smoother(xhat, xhatminus, P, Pminus)

# 예측 성능 결과값
print('예측 성능 결과값')
kalman.predict_score(data[:n], xhat_smooth.reshape(-1,))

# 미래 값 예측
forecast_steps = 6 # 미래 예측값
forward_n = 10 # 그 전 값이 얼마나 영향을 주는지 

A = kalman.state_matrix(data[n-forward_n:n] ,start=n-forward_n)
forecasts, forecast_covariances = kalman.forecast(xhat_smooth, P, A, forecast_steps)

# 미래값 성능 확인
print('미래 예측데이터와 실제 데이터 성능값')
kalman.predict_score(data[n:n+forecast_steps], forecasts)
print('-------------------------------------------------') 
# --------------------------------------------------------------

