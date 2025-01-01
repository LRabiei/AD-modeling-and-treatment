import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
import pandas as pd
from scipy.integrate import odeint
from scipy.optimize import minimize
from numpy.random import randint

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


def abeta_model(y, t,params):
    M, D, T, P, N, F = y
    l1, l2, l3, mu4, mu5, mu6,K1,K2,K3,k12, k21, mu2, mu1, hatmu1, epsilon1, k23, epsilon2, k13, mu3 = params


    dMdt = hatmu1 - k12 * M ** 2 + k21 * D - mu1 * M - k13 * M ** 3 + epsilon1 * T - k23 * M * D + epsilon2 * T
    dDdt = k12 * M ** 2 - k21 * D - mu2 * D - k23 * M * D + epsilon2 * T
    dTdt = k13 * M ** 3 - epsilon1 * T + k23 * M * D - epsilon2 * T - mu3 * T
    dPdt = l1 * D * (K1 - D) - mu4 * P
    dNdt = l2 * T * (K2 - T) - mu5 * N
    dFdt = l3 *K3* P / (K3 + N) - mu6*F

    return [dMdt, dDdt, dTdt, dPdt, dNdt, dFdt]

y0 = [5e-6, 0, 0, 0, 0,0]
theta=0.1
parameter_ranges = [
     [1e5, 1e7],  # Range for l1
     [1e3, 1e4],  # Range for l2
     [0, 1], # Range for l3
     [1e-4, 1e-3],  # Range for mu4
     [(1-theta)*1e-4, (1+theta)*1e-4],  # Range for mu5
     [4e-5, 1e-4], # Range for mu6
     [1e-07, 4e-06],# Range for K1
     [1e-06, 4e-06],# Range for K2
     [1e-03, 5e-03],# Range for K3
     [800, 1180],  # k12,
     [8e-10, 8e-10],  # k21
     [9e-09, 9e-09],  # mu2
     [1e-05, 1e-03],  # mu1
     [1e-14, 1e-14],  # hatmu1
     [1e-13, 1e-13],  # epsilon1
     [33, 43],  # k23
     [1e-11, 1e-11],  # epsilon2
     [(1-theta)*1e9, (1+theta)*1e9],  # k13
     [9e-9, 9e-09]  # mu3
]


# Specify the initial guesses for the parameters
initial_guesses = [
    (parameter_range[0] + parameter_range[1]) / 2  
    for parameter_range in parameter_ranges
]

bounds = parameter_ranges
# Define the objective function
def objective(params):
    solution = odeint(abeta_model, y0, truncated_time_points, args=(params,), rtol=1e-6, atol=1e-9)
    simulatedF= solution[:, -1]  # F is the last component
    mse = np.mean((simulatedF - truncated_observations) ** 2)
    return mse
# Minimize the sum of MSEs
result = minimize(objective, initial_guesses, method='Nelder-Mead', bounds=bounds)


parameter_names = [
    "l1", "l2","l3", "mu4", "mu5", "mu6","K1","K2","K3","k12", "k21", "mu2", "mu1", "hatmu1", "epsilon1", "k23", "epsilon2", "k13", "mu3"
]


estimated_params = result.x
for name, value in zip(parameter_names, estimated_params):
    print(f"{name} = {value}")
np.save('estimated_params',estimated_params)
simulated_solution = odeint(abeta_model, y0, truncated_time_points, args=(estimated_params,))

simulated_F=simulated_solution[:, -1]

absolute_error = np.abs(simulated_F - truncated_observations)

plt.plot(truncated_time_points, truncated_observations, label='Observed Data', linestyle='-', color='blue', linewidth=2.5)
plt.plot(truncated_time_points, simulated_F, label='Simulated F', linestyle='-', color='red')
plt.xlabel('Time (hours)')
plt.ylabel('F(t)')
plt.legend()
plt.grid(True)
plt.title('Simulated vs. Observed Data (F)')
plt.show()

