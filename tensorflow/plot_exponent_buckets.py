import sys
import re
import numpy as np
import matplotlib.pyplot as plt

def extract_buckets(lines):
    bucket_dict = {}
    for line in lines:
        match = re.search(r'(\w+)\s+\=\s+\[([\w,]+)\]', line)
        if match:
            title = match.group(1)
            buckets = match.group(2)
            buckets = buckets.split(',')
            buckets = buckets[:-1]
            buckets = [int(x) for x in buckets]
            if title not in bucket_dict:
                bucket_dict[title] = buckets
    return bucket_dict

def compute_percentile_range(buckets, percentile):
    '''
    Calculate the index of the bucket corresponding to the percentile
    '''
    histogram = np.array(buckets)
    total =  histogram.sum()
    cdf = np.cumsum(histogram / total)
    idx = np.searchsorted(cdf, percentile / 100)
    return idx

def plot_histogram(title, buckets):
    plt.figure(figsize=(20,12))
    plt.title(title)
    x = list(range(-24, 16))
    x = ['zero'] + [str(i) for i in x] + ['inf', 'nan']
    low_idx = compute_percentile_range(buckets, 0.001)
    high_idx = compute_percentile_range(buckets, 99.999)
    colors = ['orange'] * (low_idx+1)  + ['blue'] * (high_idx-low_idx-1) + ['orange'] * (len(buckets)-high_idx)
    plt.bar(x, buckets, color=colors)
    plt.xlabel('power of two')
    plt.ylabel('bucket counts')
    plt.savefig(title+'.png')


if __name__ == '__main__':
    with open(sys.argv[1]) as fi:
        lines = fi.readlines()
        bucket_dict = extract_buckets(lines)
        for title, buckets in bucket_dict.items():
            plot_histogram(title, buckets)

    
