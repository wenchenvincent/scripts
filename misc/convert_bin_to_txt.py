import argparse
import numpy as np

parser = argparse.ArgumentParser(description='This script converts binary float file generated from MIOpenDriver to text file.')
parser.add_argument('input', action='store', type=str, help='.bin file containing binary float values')
parser.add_argument('--datatype', action='store', default=0, type=int, help='1 - fp16, 2 - fp64, 0 default - fp32') 
args = parser.parse_args()
with open(args.input, "rb") as f:
  if args.datatype == 1:
    data = np.fromfile(f, np. float16)
  elif args.datatype == 2:
    data = np.fromfile(f, np. float64)
  else:
    data = np.fromfile(f, np. float32)
  print("size ", data.size)
  print(data)
  np.savetxt(args.input+".txt", data, fmt='%.8f', delimiter="\n")
