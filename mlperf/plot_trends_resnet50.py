import sys
import re
import numpy as np
import matplotlib.pyplot as plt
import functools

def get_loss_and_accuracy(lines, train=True):
    '''
    Returns an np array, first column is step, second column is loss and third column is accuracy
    '''
    if train:
        pattern = re.compile(r'^step:\s+(\d+)\s+.*\{\'train_loss\':\s*([\d\.]+),\s*\'train_accuracy\':\s*([\d\.]+)\}')
    else: #test
        pattern = re.compile(r'^step:\s+(\d+)\s+.*\{\'test_loss\':\s*([\d\.]+),\s*\'test_accuracy\':\s*([\d\.]+),')
    array = []
    for line in lines:
        match = pattern.match(line)
        if match:
            array.append([float(match.group(1)), float(match.group(2)), float(match.group(3))])
    return np.array(array)

def get_global_batch_size(lines):
    found = False
    for line in lines:
        match = re.search(r'flags_obj\.batch_size = (\d+)', line)
        if match:
            global_batch_size = int(match.group(1))
            found = True
            break
    if not found:
        print("Cannot find global batch size. Please examine the log to see if there is any issue.")
        sys.exit(1)
    return global_batch_size

def get_batch_from_step(step, global_batch_size, num_of_training_samples=1281167):
    steps_per_epoch = num_of_training_samples / global_batch_size + 1
    epoch = step / steps_per_epoch
    return epoch

def plot_trends(train_metrics, test_metrics, legend, fmt='.-b'):
    fig, ax = plt.subplots(2,2,figsize=(10,7))
    fig.suptitle("MLPerf Resnet50")
    ax[0][0].plot(train_metrics[:,0], train_metrics[:,2], fmt)
    ax[0][0].set_xlabel('Epoch')
    ax[0][0].set_ylabel('Training Accuracy')
    ax[0][0].grid()
    ax[0][0].legend([legend])


    ax[0][1].plot(train_metrics[:,0], train_metrics[:,1], fmt)
    ax[0][1].set_xlabel('Epoch')
    ax[0][1].set_ylabel('Training Loss')
    ax[0][1].grid()
    ax[0][1].legend([legend])

    ax[1][0].plot(test_metrics[:,0], test_metrics[:,2], fmt)
    ax[1][0].set_xlabel('Epoch')
    ax[1][0].set_ylabel('Test Accuracy')
    ax[1][0].grid()
    ax[1][0].legend([legend])

    ax[1][1].plot(test_metrics[:,0], test_metrics[:,1], fmt)
    ax[1][1].set_xlabel('Epoch')
    ax[1][1].set_ylabel('Test Loss')
    ax[1][1].grid()
    ax[1][1].legend([legend])


    fig.savefig('trends.png')


if __name__ == '__main__':
    with open(sys.argv[1]) as fi:
        lines = fi.readlines()
        train_metrics = get_loss_and_accuracy(lines)
        test_metrics = get_loss_and_accuracy(lines,train=False)
        global_batch_size = get_global_batch_size(lines)
        get_batch_from_step_func = functools.partial(get_batch_from_step, global_batch_size=global_batch_size)
        train_metrics[:,0] = np.vectorize(get_batch_from_step_func)(train_metrics[:,0])
        test_metrics[:,0] = np.vectorize(get_batch_from_step_func)(test_metrics[:,0])
        plot_trends(train_metrics,test_metrics, sys.argv[2])

