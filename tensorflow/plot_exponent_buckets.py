import sys
import re
import matplotlib.pyplot as plt

def extract_buckets(lines):
    bucket_dict = {}
    for line in lines:
        #match = re.search(r'(\w+)\s+\=\s+\[(\w+)\]', line)
        match = re.search(r'(\w+)\s+\=\s+\[([\w,]+)\]', line)
        if match:
            print(line)
            title = match.group(1)
            buckets = match.group(2)
            buckets = buckets.split(',')
            buckets = buckets[:-1]
            buckets = [int(x) for x in buckets]
            if title not in bucket_dict:
                bucket_dict[title] = buckets
    return bucket_dict


def plot_histogram(title, buckets):
    plt.figure(figsize=(20,12))
    plt.title(title)
    x = list(range(-24, 16))
    x = ['zero'] + [str(i) for i in x] + ['inf', 'nan']
    plt.bar(x, buckets)
    plt.xlabel('power of two')
    plt.ylabel('bucket counts')
    plt.savefig(title+'.png')


if __name__ == '__main__':
    with open(sys.argv[1]) as fi:
        lines = fi.readlines()
        bucket_dict = extract_buckets(lines)
        for title, buckets in bucket_dict.items():
            plot_histogram(title, buckets)

    
