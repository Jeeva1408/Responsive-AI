import numpy as np

salaries = np.array([50000, 60000, 1000000, 55000]) 

def add_laplace_noise (data, sensitivity, epsilon):
beta sensitivity / epsilon
noise np.random.laplace(0, beta, len(data))
return data + noise

sensitive avg = np.mean (salaries)
private avg = np.mean (add laplace noise (salaries, sensitivity=1000000, epsilon=1))
print (f"True Average: (sensitive avg)")
print (f"Private Average: {private avg}")
