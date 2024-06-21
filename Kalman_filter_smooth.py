# date: 2024-06-14
# hyuna
# Kalman Filter/Smooth

import pandas as pd
import numpy as np


class Kalman:
    def __init__(self, data, H=np.array([[1]]), R=1, Q=np.diag([0.5])):
        self.data = data
        self.z = self.make_data()
        self.A = self.state_matrix(self.data) # 상태 전이 행렬
        # self.A = np.array([[1]]) # 상태 전이 행렬
        self.H = H # 관측행렬
        self.R = R # 측정분산
        self.Q = Q # 과정분산
        self.ts_length = len(self.z)
        self.dim_state = self.Q.shape[0]
    
    # 기본 데이터 구성
    def make_data(self):
        true_data = np.array(self.data)
        observed_data = true_data + np.random.normal(0, 0.5, len(true_data))
        return observed_data
    
    # 상태 전이 행렬 구하기
    # 선형회귀를 사용하여 구할수도 있음(추후 예정)
    def state_matrix(self, data ,start=0):
        a_list = []
        for i in range(len(data)):
            try:
                a = data.loc[start+i+1] /data.loc[start+i]
                a_list.append(a)
            except:
                pass
        A = np.array([[np.mean(a_list)]])
        return A
    
    # 칼만 필터
    def kalman_filter(self):
        ts_length = self.ts_length
        dim_state = self.dim_state
        
        # 수식 필요한 값들
        xhatminus = np.zeros((ts_length, dim_state)) # 예측된 상태 추정값
        xhat = np.zeros((ts_length, dim_state)) # 필터링된 상태 추정값
        Pminus = np.zeros((ts_length, dim_state, dim_state)) # 예측된 오차 공분산
        P = np.zeros((ts_length, dim_state, dim_state)) # 필터링된 오차 공분산
        K = np.zeros((ts_length, dim_state))  # Kalman gain
        
        # 초기 값 설정
        init = self.z[0]
        xhat[0, :] = init
        xhatminus[0, :] = init
        P[0, :, :] = np.eye(dim_state)
        
        # 시간 갱신
        for k in range(1, ts_length):
            # 예측단계(Prediction Step)
            xhatminus[k,:] = self.A @ xhat[k-1, :]
            Pminus[k, :, :] = self.A @ P[k-1, :, :] @ self.A.T + self.Q
            
            # 보정단계(Correction Step)
            K[k,:] = Pminus[k, :, :] @ self.H.T @ np.linalg.inv(self.H @ Pminus[k, : ,:] @ self.H.T + self.R)
            xhat[k,:] = xhatminus[k,:] + K[k, :] @ (self.z[k] - self.H @ xhatminus[k,:])
            P[k, :, :] = (np.eye(dim_state) - K[k, :][: np.newaxis] @ self.H[np.newaxis, :]) @ Pminus[k, :, :]
        
        return xhat, xhatminus, P, Pminus
    
    # 칼만 스무딩
    def kalman_smoother(self, xhat, xhatminus, P, Pminus):
        ts_length = self.ts_length
        dim_state = self.dim_state
        
        xhat_smooth = np.zeros((ts_length, dim_state))
        P_smooth = np.zeros((ts_length, dim_state, dim_state))
        J = np.zeros((dim_state, dim_state)) # 스무딩 이득
        
        # 초기화: 스무딩 단계에서는 마지막 필터링 결과를 그대로 사용
        xhat_smooth[-1, :] = xhat[-1, :]
        P_smooth[-1, :, :] = P[-1, :, :]
        
        # 역방향 시간 갱신(Backward Pass/후방 패스)
        for k in range(ts_length-2, -1, -1):
            J = P[k, :, :] @ self.A.T @ np.linalg.inv(Pminus[k+1, :, :]) # 스무딩 이득 계산
            xhat_smooth[k, :] = xhat[k,:] + J @ (xhat_smooth[k+1, :] - xhatminus[k+1, :])
            P_smooth[k, :, :] = P[k, :, :] + J @ (P_smooth[k+1, :, :] - Pminus[k+1, :, :]) @ J.T
        
        return xhat_smooth, P_smooth
    

    # 예측
    def forecast(self, xhat, P, A, steps):
        dim_state = xhat.shape[1]
        forecasts = np.zeros((steps, dim_state))
        forecast_covariances = np.zeros((steps, dim_state, dim_state))
        
        forecasts[0, :] = xhat[-1, :]
        forecast_covariances[0, :, :] = P[-1, :, :]
        
        for k in range(1, steps):
            forecasts[k, :] = A @ forecasts[k-1, :]
            forecast_covariances[k, :, :] = A @ forecast_covariances[k-1, :, :] @ A.T + self.Q
        
        return forecasts, forecast_covariances
    
    # 예측 성능
    def predict_score(self, data, predict):
        true = np.array(data)
        mse = np.mean((true-predict)**2)
        mae = np.mean(abs(true-predict))
        
        print('MSE', mse)
        print('MAE', mae)
