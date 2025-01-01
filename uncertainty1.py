import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
import pandas as pd
from scipy.interpolate import interp1d
from scipy.integrate import odeint
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
estimated_params=np.load('estimated_params.npy')

k21 = estimated_params[10]
mu2 = estimated_params[11]
hatmu1 = estimated_params[13]
epsilon1 = estimated_params[14]
epsilon2 = estimated_params[16]
mu3 = estimated_params[18]
# Load the eigenvector matrices
U2 = np.load('noniden_eigenvectors.npy')
U2_T_pinv = np.linalg.pinv(U2.T)


parameters_array= np.load('estimated_params2.npy')


def system(y, t, params):
    M, D, T, P, N, F = y
    l1, l2, l3, mu4, mu5, mu6, K1,K2, K3, k12, mu1, k23, k13 = params
    dMdt = hatmu1 - k12 * M**2 + k21 * D - mu1 * M - k13 * M**3 + epsilon1 * T - k23 * M * D + epsilon2 * T
    dDdt = k12 * M**2 - k21 * D - mu2 * D - k23 * M * D + epsilon2 * T
    dTdt = k13 * M**3 - epsilon1 * T + k23 * M * D - epsilon2 * T - mu3 * T
    dPdt = l1 * D * (K1 - D) - mu4 * P
    dNdt = l2 * T * (K2 - T) - mu5 * N
    dFdt = (l3 *K3* P )/(K3 + N) - mu6 * F
    return [dMdt, dDdt, dTdt, dPdt, dNdt, dFdt]


sigma_array = np.array([1e-4, 6e-4, 1e-4,1e-4,1e-4,1e-4,1e-4,1e-4,1e-4])
solutions_F = []
for i in range(1000):
    epsilon = np.random.normal(0, 0.05) 
    y0 = [5e-6, 0, 0, 0, 0, epsilon]
    EPS = np.array([np.random.normal(0, sigma) for sigma in sigma_array])
    new_parameters = parameters_array * (1 + U2_T_pinv @ EPS)
    sol=odeint(system, y0,  truncated_time_points, args=(new_parameters,))

    solutions_F.append(sol[:, -1])
solutions_F_array = np.array(solutions_F)
PI_lower = np.percentile(solutions_F_array, 2.5, axis=0)
PI_upper = np.percentile(solutions_F_array, 97.5, axis=0)
PI_width = PI_upper - PI_lower
PI_lower = np.maximum(PI_lower, 0)
time_hours = truncated_time_points/ 3600
simulated_F=np.load('simulated_F.npy')
plt.figure(figsize=(9, 6))
plt.fill_between(time_hours, PI_lower, PI_upper, color='#90EE90', alpha=0.3, label='95% Prediction Interval')
plt.scatter(time_hours, truncated_observations, label='Observation 1', color='black', marker='o', s=1)
plt.plot(time_hours, simulated_F, label='Simulated F', linestyle='-', color='red',linewidth=2.5)
plt.xlabel('Time',fontsize=14)
plt.ylabel('F concentration',fontsize=14)
plt.tick_params(axis='both', which='major', labelsize=14)
plt.title('M(0)=5$\mu$M', fontsize=16)
plt.show()

