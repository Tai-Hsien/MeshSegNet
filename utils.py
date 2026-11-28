import numpy as np
import os


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def get_avail_gpu():
    '''
    works for linux
    '''
    result = os.popen("nvidia-smi").readlines()

    try:
        for i in range(len(result)):
            if 'Processes' in result[i]:
                process_idx = i

        num_gpu = 0
        for i in range(process_idx + 1):
            if 'MiB' in result[i]:
                num_gpu += 1
        gpu_list = list(range(num_gpu))

        for i in range(process_idx, len(result)):
            if result[i][22] == 'C':
                gpu_list.remove(int(result[i][5]))

        return gpu_list[0]
    except:
        print('no gpu available, return 0')
        return 0
