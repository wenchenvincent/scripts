import sys
import re

def get_tests_status(log_path, csv_path):
    #passed_pattern = r'([\/\:\w]+)\s+.+'
    passed_pattern = r'(//tensorflow/[\/\:\w]+)\s+.+(PASSED).+\s+in\s+([\d\.]+s)'
    failed_pattern = r'(//tensorflow/[\/\:\w]+)\s+.+(FAILED).+\s+in\s+([\d\.]+s)'
    timeout_pattern = r'(//tensorflow/[\/\:\w]+)\s+.+(TIMEOUT).+\s+in\s+([\d\.]+s)'
    test_status_dict = {}
    with open(log_path) as fi:
        for line in fi:
            #line = '//tensorflow/stream_executor/cuda:memcpy_test                            [0m[32mPASSED[0m in 0.1s'
            match = re.search(passed_pattern, line)
            if match:
                #print(match.group(1)+','+match.group(2)+','+match.group(3))
                test_name, status, test_time =  match.group(1, 2, 3)
                if test_name in test_status_dict:
                    if test_status_dict[test_name][0] == 'FAILED':
                        print('Warning: %s passed and failed in different runs' %(test_name))
                test_status_dict[test_name] = (status, test_time)

            match = re.search(failed_pattern, line)
            if match:
                #print(match.group(1)+','+match.group(2)+','+match.group(3))
                test_name, status, test_time =  match.group(1, 2, 3)
                if test_name in test_status_dict:
                    if test_status_dict[test_name][0] == 'PASSED':
                        print('Warning: %s passed and failed in different runs' %(test_name))
                        continue
                test_status_dict[test_name] = (status, test_time)
            match = re.search(timeout_pattern, line)
            if match:
                test_name, status, test_time =  match.group(1, 2, 3)
                if not test_name in test_status_dict:
                    test_status_dict[test_name] = (status, test_time)
                else:
                    if test_status_dict[test_name][0] == 'TIMEOUT':
                        prev_test_time = float(test_status_dict[test_name][1][:-1])
                        current_test_time = float(test_time[:-1])
                        if prev_test_time < current_test_time:
                            test_status_dict[test_name] = (status, test_time)


                #print(match.group(1)+','+match.group(2)+','+match.group(3))
    with open(csv_path, 'w') as fo:
        fo.write('test, status, time\n')
        for test_name, v in test_status_dict.items():
            fo.write('%s,%s,%s\n' %(test_name, v[0], v[1]) )

if __name__ == '__main__':
    get_tests_status(sys.argv[1], sys.argv[2])
    
