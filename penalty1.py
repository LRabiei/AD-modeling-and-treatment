import pandas as pd
import numpy as np
from scipy.integrate import odeint
from SALib.sample import saltelli
from SALib.analyze import sobol
from joblib import Parallel, delayed
import multiprocessing
import matplotlib.pyplot as plt
import sympy as sp
from scipy.optimize import minimize


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
U2 = np.load('noniden_eigenvectors.npy')
U1=np.load('iden_eigenvectors.npy')
U=np.load('first_order_Soboleigenvectors.npy')
estimated_params=np.load('estimated_params.npy')

k21 = estimated_params[10]
mu2 = estimated_params[11]
hatmu1 = estimated_params[13]
epsilon1 = estimated_params[14]
epsilon2 = estimated_params[16]
mu3 = estimated_params[18]
estimated_params1 = np.array([
     estimated_params[0],   # Range for l1
     estimated_params[1],  # Range for l2
     estimated_params[2],   # Range for l3
     estimated_params[3],  # Range for mu4
     estimated_params[4],   # Range for mu5
     estimated_params[5],   # Range for mu6
     estimated_params[6],  # Range for K1
     estimated_params[7],   # Range for K1
     estimated_params[8],   # Range for K3
     estimated_params[9],   # k12
     estimated_params[12],   # mu1
     estimated_params[15],   # k23
     estimated_params[17],  # k13
])

Test=U2.T@estimated_params1

y0 = [5e-6, 0, 0, 0, 0, 0]
def abeta_model(y, t,params):
    M, D, T, P, N, F = y
    l1, l2, l3, mu4, mu5, mu6,K1,K2,K3,k12, mu1, k23, k13 = params


    dMdt = hatmu1 - k12 * M ** 2 + k21 * D - mu1 * M - k13 * M ** 3 + epsilon1 * T - k23 * M * D + epsilon2 * T
    dDdt = k12 * M ** 2 - k21 * D - mu2 * D - k23 * M * D + epsilon2 * T
    dTdt = k13 * M ** 3 - epsilon1 * T + k23 * M * D - epsilon2 * T - mu3 * T
    dPdt = l1 * D * (K1 - D) - mu4 * P
    dNdt = l2 * T * (K2 - T) - mu5 * N
    dFdt = (l3 *K3* P )/(K3 + N) - mu6 * F

    return [dMdt, dDdt, dTdt, dPdt, dNdt, dFdt]

theta=0.2
parameter_ranges = [
     [(1-theta)*estimated_params1[0], (1+theta)*estimated_params1[0]],  # Range for l1
    [(1-theta)*estimated_params1[1], (1+theta)*estimated_params1[1]],  # Range for l2
    [(1-theta)*estimated_params1[2], (1+theta)*estimated_params1[2]], # Range for l3
     [(1-theta)*estimated_params1[3], (1+theta)*estimated_params1[3]],  # Range for mu4
    [(1-theta)*estimated_params1[4], (1+theta)*estimated_params1[4]],  # Range for mu5
    [(1-theta)*estimated_params1[5], (1+theta)*estimated_params1[5]],# Range for mu6
    [(1-theta)*estimated_params1[6], (1+theta)*estimated_params1[6]],# Range for K1
     [(1-theta)*estimated_params1[7], (1+theta)*estimated_params1[7]],# Range for K2
     [(1-theta)*estimated_params1[8], (1+theta)*estimated_params1[8]],# Range for K3
     [(1-theta)*estimated_params1[9], (1+theta)*estimated_params1[9]], # k12,
     [(1-theta)*estimated_params1[10], (1+theta)*estimated_params1[10]],  # mu1
     [(1-theta)*estimated_params1[11], (1+theta)*estimated_params1[11]], # k23
     [(1-theta)*estimated_params1[12], (1+theta)*estimated_params1[12]], # k13
]

initial_guesses = [
    (parameter_range[0] + parameter_range[1]) / 2 
    for parameter_range in parameter_ranges
]

bounds = parameter_ranges

def objective(params):
    solution = odeint(abeta_model, y0,  truncated_time_points, args=(params,))
    simulatedF = solution[:, -1]  
    relative_difference = (params - estimated_params1)/ estimated_params1 
    mse = np.mean((simulatedF - truncated_observations) ** 2) + 10* np.mean((U2.T @ relative_difference[:, np.newaxis]) ** 2)
    return mse

result = minimize(objective, initial_guesses, method='Nelder-Mead', bounds=bounds)

parameter_names = [
    "l1", "l2","l3", "mu4", "mu5", "mu6","K1","K2","K3","k12",  "mu1",  "k23",  "k13"
]

estimated_params2 = result.x
for name, value in zip(parameter_names, estimated_params2):
 print(f"{name} = {value}")


TV1=objective(estimated_params1)
print("value1:",TV1)  
TV2=objective(estimated_params2)
print("value1:",TV2)
rel=(estimated_params2 - estimated_params1) / estimated_params1
print("error:",rel) 
penalty=np.mean((U2.T @rel) ** 2)
print("penalty:",penalty) 



simulated_solution = odeint(abeta_model, y0,  truncated_time_points, args=(estimated_params2,))

simulated_F=simulated_solution[:, -1]
absolute_error = np.abs(simulated_F - truncated_observations)
np.save('simulated_F', simulated_F)
np.save('estimated_params2', estimated_params2)
np.save('parameters_array1.npy', estimated_params2)
simulated_M=simulated_solution[:, 0]
simulated_D=simulated_solution[:, 1]
simulated_T=simulated_solution[:, 2]
simulated_P=simulated_solution[:, 3]
simulated_N=simulated_solution[:, 4]
np.save('simulated_M', simulated_M)
np.save('simulated_D', simulated_D)
np.save('simulated_T', simulated_T)
np.save('simulated_P', simulated_P)
np.save('simulated_N', simulated_N)

plt.plot( truncated_time_points,truncated_observations, label='Observed Data', linestyle='-', color='blue', linewidth=2.5)

plt.plot( truncated_time_points, simulated_F, label='Simulated F', linestyle='-', color='red')
plt.xlabel('Time (hours)')
plt.ylabel('F(t)')
plt.legend()
plt.grid(True)
plt.title('Simulated vs. Observed Data (F)')
plt.show()
plt.plot( truncated_time_points, absolute_error, label='Absolute Error', linestyle='-', color='purple')
plt.xlabel('Time (hours)')
plt.ylabel('Absolute Error')
plt.legend()
plt.grid(True)
plt.title('Absolute Error between Simulated F and Observed Data')
plt.show()



