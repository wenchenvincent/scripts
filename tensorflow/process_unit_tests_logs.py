import sys
import re

def get_tests_status(log_path):
    #passed_pattern = r'([\/\:\w]+)\s+.+'
    passed_pattern = r'(//tensorflow/[\/\:\w]+)\s+.+(PASSED).+'
    failed_pattern = r'(//tensorflow/[\/\:\w]+)\s+.+(FAILED).+'
    timeout_pattern = r'(//tensorflow/[\/\:\w]+)\s+.+(TIMEOUT).+'
    with open(log_path) as fi:
        for line in fi:
            #line = '//tensorflow/stream_executor/cuda:memcpy_test                            [0m[32mPASSED[0m in 0.1s'
            #line = '//tensorflow/stream_executor/cuda:redzone_allocator_test_gpu'
            match = re.search(passed_pattern, line)
            if match:
                print(match.group(1)+','+match.group(2))
            match = re.search(failed_pattern, line)
            if match:
                print(match.group(1)+','+match.group(2))
            match = re.search(timeout_pattern, line)
            if match:
                print(match.group(1)+','+match.group(2))

if __name__ == '__main__':
    get_tests_status(sys.argv[1])
    
