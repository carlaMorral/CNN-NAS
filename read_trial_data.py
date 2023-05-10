import sys
import os
from datetime import datetime

class Trial:
    def __init__(self, tid, start, end, metric):
        self.id = tid
        self.start = datetime.strptime(start, '[%Y-%m-%d %H:%M:%S]')
        self.end = datetime.strptime(end, '[%Y-%m-%d %H:%M:%S]')
        self.metric = metric

    def __lt__(self, other):
        return self.end < other.end

if len(sys.argv) < 2:
    print("Please specify an experiment ID")
    exit()

exp_id = sys.argv[1]
trials = []
for trial in os.listdir(f'../nni-experiments/{exp_id}/trials'):
    with open(f'../nni-experiments/{exp_id}/trials/{trial}/trial.log', 'r') as f:
        lines = f.readlines()
    fl = lines[0].rstrip().split()
    start = ' '.join(fl[0:2])
    ll = lines[-1].rstrip().split()
    end = ' '.join(ll[0:2])
    metric = None
    if len(ll) >= 6:
        if ll[3] == 'Final':
            metric = float(ll[5])
    if metric:
        trials.append(Trial(trial, start, end, metric))

trials.sort()
lowest_start = trials[0].start
best_id = None
best_metric = 0
for trial in trials:
    if trial.start < lowest_start:
        lowest_start = trial.start
    if trial.metric > best_metric:
        best_metric = trial.metric
        best_id = trial.id

print(f'Info for experiment {exp_id}:')
print(f'Best model ID: {best_id}')
print(f'Best metric: {best_metric}')
if len(trials) >= 250:
    print(f'Time for 250 trials: {str(trials[249].end - lowest_start)}')
print(f'Time for {len(trials)} trials: {str(trials[-1].end - lowest_start)}')
