import numpy as np
import pandas as pd
from scipy.integrate import odeint
import matplotlib.pyplot as plt
import math


path = r'C:\AD project\data_excel.xlsx'
df = pd.read_excel(path)
averages_df = pd.DataFrame()
averages_df['time / h'] = df['time / h']
num_initial_conditions = 10
for i in range(num_initial_conditions):
    start_col = 1 + i * 4  
    end_col = start_col + 4  
    group_name = f'Average_{i+1}'  
    averages_df[group_name] = df.iloc[:, start_col:end_col].mean(axis=1)
time_column = averages_df.iloc[:, 0]
normalized_columns = averages_df.iloc[:, 1:].apply(lambda col: (col - col.min()) / (col.max() - col.min()), axis=0)
normalized_df = pd.concat([time_column, normalized_columns], axis=1)
time_points =  normalized_df['time / h'].values
observations = normalized_df['Average_1'].values
zero_index = np.where(observations == 0)[0][0]
one_index = zero_index + np.where(observations[zero_index:] == 1)[0][0]
truncated_observations = observations[zero_index:]
truncated_observations[one_index - zero_index:] = 1
truncated_time_points = 3600*time_points[zero_index:]
adjusted_time_points =( truncated_time_points - truncated_time_points[0])
lambda1 = 5457223.569449566
lambda2 =5634.398747390245
lambda3 = 0.2986572660854546
mu4 = 0.00040520011505642195
mu5 = 9.560003590902461e-05
mu6 = 6.919398305707606e-05
K1 = 1.2601885934893833e-06
K2 = 4.0009167517170816e-06
K3 = 0.0033380133215667324
k12 = 989.7542658802313
mu1 = 0.0005589991956551556
k23 = 33.19716600876461
k13 = 1039460056.6920896
k21 = 8e-10
mu2 = 9e-9
hat_mu_1 = 1e-14
epsilon1 = 1e-13
epsilon2 = 1e-11
mu3 = 9e-9
j1 = 1
j2 = 1

Fsh = np.load('simulated_F.npy')
y0 = [5e-6, 0, 0, 0, 0, 0]
optimal_alpha_D, optimal_beta_D, optimal_cE_D, U_D_max = 2, 2e-4, 1,  (43 / 7) * mu6
optimal_alpha_L, optimal_beta_L, optimal_cE_L, U_L_max = 2, 2e-4, 1, 0.16 * mu6
optimal_alpha_A, optimal_beta_A, optimal_cE_A, U_A_max = 2, 4e-4, 0.1,0.182 * mu6

def g_t_exponential(t, alpha, beta):
    return alpha * np.exp(-beta * t)
s
def state_eqs(y, t, u_D, u_L, u_A, truncated_time_points):
    M, D, T, P, N, F = y
    u_D_t = np.interp(t, truncated_time_points, u_D)
    u_L_t = np.interp(t, truncated_time_points, u_L)
    u_A_t = np.interp(t, truncated_time_points, u_A)
    dMdt = hat_mu_1 - k12 * M**2 + k21 * D - k13 * M**3 + epsilon1 * T - k23 * M * D + epsilon2 * T - mu1 * M
    dDdt = k12 * M**2 - k21 * D - k23 * M * D + epsilon2 * T - mu2 * D
    dTdt = k13 * M**3 + k23 * M * D - epsilon1 * T - epsilon2 * T - mu3 * T
    dPdt = lambda1 * D * (K1 - D) - mu4 * P - u_L_t * P - u_A_t * P
    dNdt = lambda2 * T * (K2 - T) - mu5 * N - u_A_t * N
    dFdt = lambda3 * (P / (1 + N / K3)) - mu6 * F - u_D_t * F - u_L_t * F - u_A_t * F
    return [dMdt, dDdt, dTdt, dPdt, dNdt, dFdt]

def adjoint_eqs(l, t, y, u_D, u_L, u_A, truncated_time_points, cE_D, cE_L, cE_A):
    M, D, T, P, N, F = [np.interp(t, truncated_time_points, y[i]) for i in range(6)]
    lambda_M, lambda_D, lambda_T, lambda_P, lambda_N, lambda_F = l
    u_D_t = np.interp(t, truncated_time_points, u_D)
    u_L_t = np.interp(t, truncated_time_points, u_L)
    u_A_t = np.interp(t, truncated_time_points, u_A)
    dlambda_M_dt = lambda_M * (2 * k12 * M + 3 * k13 * M**2 + k23 * D + mu1) - lambda_D * (2 * k12 * M - k23 * D) - lambda_T * (3 * k13 * M**2 + k23 * D)
    dlambda_D_dt = lambda_M * (-k21 + k23 * M) + lambda_D * (k21 + k23 * M + mu2) - lambda_T * (k23 * M) - lambda_P * (lambda1 * (K1 - 2 * D))
    dlambda_T_dt = -lambda_M * (epsilon1 + epsilon2) - lambda_D * (epsilon2) + lambda_T * (epsilon1 + epsilon2 + mu3) - lambda_N * lambda2 * (K2 - 2 * T)
    dlambda_P_dt = (mu4 + u_L_t + u_A_t) * lambda_P - lambda3 * lambda_F / (1 + N / K3)
    dlambda_N_dt = -j1 + (mu5 + u_A_t) * lambda_N + lambda3 * lambda_F * P / (K3 * (1 + N / K3)**2)
    dlambda_F_dt = -j2 + (mu6 + u_D_t + u_L_t + u_A_t) * lambda_F - cE_D * 0.221 * u_D_t / U_D_max - cE_L * 0.0563 * u_L_t / U_L_max - cE_A * 0.301 * u_A_t / U_A_max
    return [dlambda_M_dt, dlambda_D_dt, dlambda_T_dt, dlambda_P_dt, dlambda_N_dt, dlambda_F_dt]

def forward_backward_sweep(alpha_D, beta_D, cE_D, alpha_L, beta_L, cE_L, alpha_A, beta_A, cE_A):
    u_D = np.zeros(len(truncated_time_points))
    u_L = np.zeros(len(truncated_time_points))
    u_A = np.zeros(len(truncated_time_points))
    
    for iteration in range(1000): 
        sol_state = odeint(state_eqs, y0, truncated_time_points, args=(u_D, u_L, u_A, truncated_time_points))
        y = sol_state.T
        l0 = [0, 0, 0, 0, 0, 1e4]
        sol_adjoint = odeint(adjoint_eqs, l0, truncated_time_points[::-1], args=(y, u_D, u_L, u_A, truncated_time_points, cE_D, cE_L, cE_A))
        l = sol_adjoint.T[:, ::-1]
        F, P, N = y[5, :], y[3, :], y[4, :]
        lambda_F, lambda_P, lambda_N = l[5, :], l[3, :], l[4, :]
        g_D = g_t_exponential(truncated_time_points, alpha_D, beta_D)
        g_L = g_t_exponential(truncated_time_points, alpha_L, beta_L)
        g_A = g_t_exponential(truncated_time_points, alpha_A, beta_A)
        u_D_new = np.minimum(U_D_max, np.maximum(0, (U_D_max**2 / g_D) * (lambda_F * F - cE_D * 0.221 * F / U_D_max)))
        u_L_new = np.minimum(U_L_max, np.maximum(0, (U_L_max**2 / g_L) * (lambda_P * P + lambda_F * F - cE_L * 0.0563 * F / U_L_max)))
        u_A_new = np.minimum(U_A_max, np.maximum(0, (U_A_max**2 / g_A) * (lambda_P * P + lambda_F * F + lambda_N * N - cE_A * 0.301 * F / U_A_max)))

        u_D = 0.5 * u_D + 0.5 * u_D_new
        u_L = 0.5 * u_L + 0.5 * u_L_new
        u_A = 0.5 * u_A + 0.5 * u_A_new

    return u_D, u_L, u_A, y

u_D_opt, u_L_opt, u_A_opt, state_variables = forward_backward_sweep(
    optimal_alpha_D, optimal_beta_D, optimal_cE_D,
    optimal_alpha_L, optimal_beta_L, optimal_cE_L,
    optimal_alpha_A, optimal_beta_A, optimal_cE_A)
F_opt = state_variables[5, :] 