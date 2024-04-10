import sys
import re

def process_logs(filename):
    pattern_device = re.compile(r'hipSetDevice \( (\d+) \)')
    pattern_create = re.compile(r'hipStreamCreateWithFlags: Returned hipSuccess : stream:(0x[0-9a-fA-F]+)')
    pattern_destroy = re.compile(r'hipStreamDestroy \( stream:(0x[0-9a-fA-F]+) \)')

    address_set = {}
    max_set_size = 0

    with open(filename, 'r') as file:
        for line in file:
            match_device = pattern_device.search(line)
            match_create = pattern_create.search(line)
            match_destroy = pattern_destroy.search(line)

            if match_device:
                device = match_device.group(1)
                if device not in address_set:
                    address_set[device] = set()

            if match_create:
                address = match_create.group(1)
                address_set[device].add(address)
                max_set_size = max(max_set_size, len(address_set[device]))
            elif match_destroy:
                address = match_destroy.group(1)
                if address in address_set[device]:
                    address_set[device].remove(address)

    return max_set_size

filename = sys.argv[1]
max_set_size = process_logs(filename)
print("Max size of the set:", max_set_size)
