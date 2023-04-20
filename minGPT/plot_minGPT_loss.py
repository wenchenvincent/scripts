import sys
import re
import numpy as np
import json

def parse_log(log_file):
    '''
    Returns an np array, first column is iter, second column is loss 
    '''
    with open(log_file, 'r') as f:
        lines = f.readlines()
                    
    pattern = re.compile(r'iter (\d+): train loss ([\d.]+)')
                                
    iters = []
    losses = []
                                    
    for line in lines:
        match = pattern.search(line)
                                                                            
        if match:
            iters.append(int(match.group(1)))
            losses.append(float(match.group(2)))
                                                                                                                                            
    return np.array([iters, losses])

def plot_trends(json_file_path):
    '''
    Read from a json file and generate plots
    
    An example json file:

    {
        "title" : "Train loss Curve",
        "plots" : {
                    "fp32" : "te_fp32.log",
                    "fp8" : "te_fp8.log"
                  },
        "output_file" : "minGPT_loss.png"
    }
    '''
    import matplotlib.pyplot as plt
    with open(json_file_path, 'r') as fi:
        data = json.load(fi)

    title = data['title']
    output_name = data['output_file']
    plots_data = data['plots']
    
    for legend in plots_data.keys():
        loss = parse_log(plots_data[legend])
        x = loss[0]
        y = loss[1]
        plt.plot(x,y, label=legend)

    plt.xlabel('Iteration')
    plt.ylabel('Training Loss')
    plt.legend()
    plt.title(title)

    plt.savefig(output_name)


if __name__ == '__main__':
    plot_trends(sys.argv[1])
