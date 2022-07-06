import numpy as np

def load_matrix_from_file(file_name, m, n, data_type, col_major=True):
    data = np.fromfile(file_name, data_type)
    if col_major:
        matrix = np.reshape(data, (n, m))
        matrix = np.transpose(matrix)
    else:
        matrix = np.reshape(data, (m, n))
    return matrix

def calculate_norm_error(C, ref_C, norm_type):
    return np.linalg.norm(C.astype(np.float64)-ref_C.astype(np.float64), ord=norm_type) / np.linalg.norm(ref_C.astype(np.float64), ord=norm_type)


def calculate_relative_error(C, ref_C):
    relative_error = np.abs(C-ref_C) / np.abs(ref_C)
    idx = np.argmax( relative_error)
    relative_error = np.amax(relative_error)

    return relative_error, idx
    


# The memory layout of the data is column major
A = load_matrix_from_file('A.dat', 512, 512, np.float16) 
B = load_matrix_from_file('B.dat', 512, 2700, np.float16)
C = load_matrix_from_file('C.dat', 512, 2700, np.float16)

ref_C = np.dot(A,B)
print('ref_C=', ref_C)
print('C=', C)

norm1 = calculate_norm_error(C, ref_C, 1)
print('norm 1=', norm1)
norm_f = calculate_norm_error(C, ref_C, 'fro')
print('Frobenius norm=', norm_f)

relative_error, idx = calculate_relative_error(C, ref_C)
y = idx // 2700
x = idx % 2700
print('relative error is', relative_error)
print('C[%d][%d]=' % (y, x), C[y][x])
print('ref_C[%d][%d]=' % (y, x), ref_C[y][x])


eps = 1.0/1024
threshold = 50
error_bound = max(512, 2700) * eps * threshold
print('error_bound=', error_bound)

#relative_error_bound = np.sqrt(2700) * eps
relative_error_bound = 2 * 2700 * eps
print('relative_error_bound=', relative_error_bound)
