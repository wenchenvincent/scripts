import numpy as np

def load_matrix_from_file(file_name, m, n, data_type, col_major=True):
    data = np.fromfile(file_name, data_type)
    if col_major:
        matrix = np.reshape(data, (n, m))
        matrix = np.transpose(matrix)
    else:
        matrix = np.reshape(data, (m, n))
    return matrix

A = load_matrix_from_file('A.dat', 512, 512, np.float16) 
B = load_matrix_from_file('B.dat', 512, 2700, np.float16)
C = load_matrix_from_file('C.dat', 512, 2700, np.float16)

ref_C = np.dot(A,B)
print(ref_C)
print(C)

norm1 = np.linalg.norm(C-ref_C, ord=1)
print('norm1=', norm1)

eps = 1.0/1024
threshold = 50
error_bound = max(512, 2700) * eps * threshold
print('error_bound=', error_bound)
