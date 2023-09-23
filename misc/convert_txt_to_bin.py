mport argparse
import numpy as np

parser = argparse.ArgumentParser(description='This script converts text file containing floats to bin file consumed by MIOpenDriver.')
parser.add_argument('input', action='store', type=str, help='.txt file containing space-seperated float values.')
parser.add_argument('--datatype', action='store', default=0, type=int, help='1 - fp16, 0 default - fp32') 
args = parser.parse_args()
with open(args.input, "r") as f:
  if args.datatype == 1:
     data = np.loadtxt(f, np.float16)
  else:
     data = np.loadtxt(f, np.float32)
  print("size ", data.size)
  print(data)
  with open(args.input+".bin","wb") as wf:
    data.tofile(wf)
