import pandas as pd
import numpy as np
from scipy.integrate import odeint
from SALib.sample import saltelli
from SALib.analyze import sobol
from joblib import Parallel, delayed
import multiprocessing
import matplotlib.pyplot as plt
import sympy as sp

# Specify the path to your Excel file
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
estimated_params=np.load('estimated_params.npy')
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

k21 = estimated_params[10]
mu2 = estimated_params[11]
hatmu1 = estimated_params[13]
epsilon1 = estimated_params[14]
epsilon2 = estimated_params[16]
mu3 = estimated_params[18]

y0 = [5e-6, 0, 0, 0, 0, 0]
def abeta_model(y, t, params):
    M, D, T, P, N, F = y
    l1, l2, l3, mu4, mu5, mu6, K1, K2, K3, k12, mu1, k23, k13= params
    
    dMdt = hatmu1 - k12 * M**2 + k21 * D - mu1 * M - k13 * M**3 + epsilon1 * T - k23 * M * D + epsilon2 * T
    dDdt = k12 * M**2 - k21 * D - mu2 * D - k23 * M * D + epsilon2 * T
    dTdt = k13 * M**3 - epsilon1 * T + k23 * M * D - epsilon2 * T - mu3 * T
    dPdt = l1 * D * (K1 - D) - mu4 * P
    dNdt = l2 * T * (K2 - T) - mu5 * N
    dFdt = (l3 *K3* P )/(K3 + N) - mu6 * F
    return [dMdt, dDdt, dTdt, dPdt, dNdt, dFdt]

theta = 10**(-10)
problem = {
    'num_vars': 13,
    'names': ['l1', 'l2', 'l3', 'mu4', 'mu5', 'mu6', 'K1', 'K2','K3', 'k12',  'mu1', 'k23', 'k13'],
    'bounds': [
    [estimated_params[0], (1 + theta) *estimated_params[0]],  # Range for l1
    [estimated_params[1], (1 + theta) *estimated_params[1]],  # Range for l2
    [estimated_params[2], (1 + theta) *estimated_params[2]],  # Range for l3
    [estimated_params[3], (1 + theta) *estimated_params[3]],  # Range for mu4
    [estimated_params[4], (1 + theta) *estimated_params[4]],  # Range for mu5
    [estimated_params[5], (1 + theta) *estimated_params[5]],  # Range for mu6
    [estimated_params[6], (1 + theta) *estimated_params[6]],  # Range for K1
    [estimated_params[7], (1 + theta) *estimated_params[7]],  # Range for K2
    [estimated_params[8], (1 + theta) *estimated_params[8]],  # Range for K3
    [estimated_params[9], (1 + theta) *estimated_params[9]],  # k12
    [estimated_params[12], (1 + theta) *estimated_params[12]],  # mu1
    [estimated_params[15], (1 + theta) *estimated_params[15]],  # k23
    [estimated_params[17], (1 + theta) *estimated_params[17]],  # k13
]

}

vals = saltelli.sample(problem, 2**10)
def solve_ode(params):
    return odeint(abeta_model, y0, truncated_time_points , args=(params,))[:, 5]

num_cores = multiprocessing.cpu_count()

results = Parallel(n_jobs=num_cores)(
    delayed(solve_ode)(params) for params in vals
)

Y = np.array(results)
first_order_sensitivity_matrix = []
for t_idx in range(1, 221):  
    F_values_at_t = Y[:, t_idx]
    sobol_indices_at_t = sobol.analyze(problem, F_values_at_t, print_to_console=False)
    first_order_sensitivity_at_t = sobol_indices_at_t['S1']
    first_order_sensitivity_matrix.append(first_order_sensitivity_at_t)

first_order_sensitivity_matrix_np = np.array(first_order_sensitivity_matrix)

first_order_sensitivity_matrix_transposed = first_order_sensitivity_matrix_np.T

first_order_ST_S_productSobol = np.dot(first_order_sensitivity_matrix_transposed, first_order_sensitivity_matrix_np)

first_order_rank_ST_SSobol = np.linalg.matrix_rank(first_order_ST_S_productSobol)
print("The rank of transpose(S).S sobol1 is:", first_order_rank_ST_SSobol)


first_order_Soboleigenvalues, first_order_Soboleigenvectors = np.linalg.eig(first_order_ST_S_productSobol)

np.save('first_order_Soboleigenvectors', first_order_Soboleigenvectors)
threshold = 1e-5
iden_indices = np.where(first_order_Soboleigenvalues > threshold)[0]

iden_eigenvalues = first_order_Soboleigenvalues[iden_indices]
iden_eigenvectors = first_order_Soboleigenvectors[:, iden_indices]
noniden_indices = np.where(first_order_Soboleigenvalues < threshold)[0]
noniden_eigenvalues = first_order_Soboleigenvalues[noniden_indices]
noniden_eigenvectors = first_order_Soboleigenvectors[:, noniden_indices]





