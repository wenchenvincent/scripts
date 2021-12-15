import argparse
import numpy as np

parser = argparse.ArgumentParser(description='This script converts binary float file generated from MIOpenDriver to text file.')
parser.add_argument('input', action='store', type=str, help='.bin file containing binary float values')
args = parser.parse_args()
with open(args.input, "rb") as f:
  data = np.fromfile(f, np. float16)
  print("size ", data.size)
  print(data)
  np.savetxt(args.input+".txt", data, fmt='%.8f', delimiter="\n")
