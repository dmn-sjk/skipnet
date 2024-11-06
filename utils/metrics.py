import yaml
import numpy as np


def save_final_metrics(metrics: dict, save_path):
    with open(save_path, 'w') as file:
        # to parse OrderedDict correctly
        yaml.safe_dump(dict(metrics), file, default_flow_style=False, sort_keys=False)

def accuracy(output, target, topk=(1,)):
    """Computes the accuracy@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


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


class ListAverageMeter(object):
    """Computes and stores the average and current values of a list"""
    def __init__(self):
        self.len = 10000  # set up the maximum length
        self.reset()

    def reset(self):
        self.val = [0] * self.len
        self.avg = [0] * self.len
        self.sum = [0] * self.len
        self.count = 0

    def set_len(self, n):
        self.len = n
        self.reset()

    def update(self, vals, n=1):
        assert len(vals) == self.len, 'length of vals not equal to self.len'
        self.val = vals
        for i in range(self.len):
            self.sum[i] += self.val[i] * n
        self.count += n
        for i in range(self.len):
            self.avg[i] = self.sum[i] / self.count

def backward_transfer_metric(task_performances, current_task_index):
    """
    Calculate the backward transfer metric for continual learning.
    
    The backward transfer metric measures how much the model's performance on previous tasks
    improves or degrades as it learns new tasks.
    
    Parameters:
    task_performances (list): A list of task performance values, where each value represents
                             the model's performance on a task. The list is ordered
                             chronologically by when the tasks were learned.
    current_task_index (int): The index of the current task being learned.
    
    Returns:
    float: The backward transfer metric value.
    """
    if current_task_index == 0:
        return 0.0
    
    previous_tasks_performances = task_performances[:current_task_index]
    current_task_performance = task_performances[current_task_index]
    
    # Calculate the average performance on previous tasks
    previous_tasks_avg = np.mean(previous_tasks_performances)
    
    # Calculate the backward transfer metric
    backward_transfer = (current_task_performance - previous_tasks_avg) / previous_tasks_avg
    
    return backward_transfer

def forgetting_metric(task_performances, current_task_index):
    """
    Calculate the forgetting metric for continual learning.
    
    The forgetting metric measures how much the model's performance on previous tasks
    degrades as it learns new tasks.
    
    Parameters:
    task_performances (list): A list of task performance values, where each value represents
                             the model's performance on a task. The list is ordered
                             chronologically by when the tasks were learned.
    current_task_index (int): The index of the current task being learned.
    
    Returns:
    float: The forgetting metric value.
    """
    if current_task_index == 0:
        return 0.0
    
    previous_tasks_performances = task_performances[:current_task_index]
    best_previous_performance = max(previous_tasks_performances)
    current_task_performance = task_performances[current_task_index]
    
    # Calculate the forgetting metric
    forgetting = (best_previous_performance - current_task_performance) / best_previous_performance
    
    return forgetting