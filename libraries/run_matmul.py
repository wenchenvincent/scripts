import argparse
import numpy as np


def matmul(a_file, b_file, out_file, trans_a, trans_b, m, n, k, column_major=True):
    with open(a_file, "rb") as f:
        if args.datatype == 1:
            a = np.fromfile(f, np. float16)
        elif args.datatype == 2:
            a = np.fromfile(f, np. float64)
        else:
            a = np.fromfile(f, np. float32)
    
    with open(b_file, "rb") as f:
        if args.datatype == 1:
            b = np.fromfile(f, np. float16)
        elif args.datatype == 2:
            b = np.fromfile(f, np. float64)
        else:
            b = np.fromfile(f, np. float32)

    if column_major:
        A = a.reshape(k, m).T
        B = b.reshape(n, k).T
    else:
        A = a.reshape(m, k)
        B = b.reshape(k, n)
    if trans_a:
        A = A.T
    if trans_b:
        B = B.T
    print(A.shape)
    print(B.shape)
    if column_major:
        result = np.matmul(A, B).T
    else:
        result = np.matmul(A, B)
    with open(out_file, "wb") as wf:
        result.tofile(wf)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('a_file', action='store', type=str, help='.bin file containing binary float values for a')
    parser.add_argument('b_file', action='store', type=str, help='.bin file containing binary float values for b')
    parser.add_argument('out_file', action='store', type=str, help='.bin file containing binary float values for d')
    parser.add_argument('--trans_a', action='store_true', default=False, help='bool value of transA')
    parser.add_argument('--trans_b', action='store_true', default=False, help='bool value of transB')
    parser.add_argument('--datatype', action='store', default=0, type=int, help='1 - fp16, 2 - fp64, 0 default - fp32') 
    parser.add_argument('--m', action='store', default=0, type=int, help='m in GEMM') 
    parser.add_argument('--n', action='store', default=0, type=int, help='n in GEMM') 
    parser.add_argument('--k', action='store', default=0, type=int, help='k in GEMM') 
    parser.add_argument('--column_major', action='store_true', default=False, help='bool value of column major')
    args = parser.parse_args()
    matmul(args.a_file, args.b_file, args.out_file, args.trans_a, args.trans_b, args.m, args.n, args.k, args.column_major)
    
    


    
