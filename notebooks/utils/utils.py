import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

figsize=(9,3)
def generate_RUL_data(data):
    # Generate RUL data
    cnts = data.groupby('machine')[['cycle']].count()
    cnts.columns = ['ftime']
    tmp = data.join(cnts, on='machine')
    rul = tmp['ftime'] - tmp['cycle']
    return rul

def load_cmapss_data(path: str, pattern: str):
    cols = ['machine', 'cycle', 'p1', 'p2', 'p3'] + [f's{i}' for i in range(1, 22)]
    
    nmcn = 0
    data_list = []
    for entry in Path(path).glob(pattern):
        # Read data
        data = pd.read_csv(entry, sep=' ', header=None, usecols=range(0, len(cols)), names=cols) 
        
        # Add the data source
        data.insert(0, "src", entry.stem)
        # Shift the machine numbers
        data['machine'] += nmcn
        nmcn += len(data['machine'].unique()) 
        
        # Generate RUL data
        data['rul'] = generate_RUL_data(data)
        data_list.append(data)
    
    
    data = pd.concat(data_list)

    return data

def split_by_field(data, field):
    res = {}
    for fval, gdata in data.groupby(field):
        res[fval] = gdata
    return res

def partition_by_machine(data, tr_machines):
    # Separate
    tr_machines = set(tr_machines)
    tr_list, ts_list = [], []
    for mcn, gdata in data.groupby('machine'):
        if mcn in tr_machines:
            tr_list.append(gdata)
        else:
            ts_list.append(gdata)
    # Collate again
    tr_data = pd.concat(tr_list)
    if len(ts_list) > 0:
        ts_data = pd.concat(ts_list)
    else:
        ts_data = pd.DataFrame(columns=tr_data.columns)
    return tr_data, ts_data

def plot_rul(pred=None, target=None,
        stddev=None,
        q1_3=None,
        same_scale=True,
        figsize=figsize, autoclose=True):
    if autoclose:
        plt.close('all')
    plt.figure(figsize=figsize)
    if target is not None:
        plt.plot(range(len(target)), target, label='target',
                color='tab:orange')
    if pred is not None:
        if same_scale or target is None:
            ax = plt.gca()
        else:
            ax = plt.gca().twinx()
        ax.plot(range(len(pred)), pred, label='pred',
                color='tab:blue')
        if stddev is not None:
            ax.fill_between(range(len(pred)),
                    pred-stddev, pred+stddev,
                    alpha=0.3, color='tab:blue', label='+/- std')
        if q1_3 is not None:
            ax.fill_between(range(len(pred)),
                    q1_3[0], q1_3[1],
                    alpha=0.3, color='tab:blue', label='1st/3rd quartile')
    plt.legend()
    plt.tight_layout()

def opt_threshold_and_plot(machine, pred, th_range, cmodel,
        plot=True, figsize=figsize, autoclose=True):
    # Compute the optimal threshold
    costs = [cmodel.cost(machine, pred, thr) for thr in th_range]
    opt_th = th_range[np.argmin(costs)]
    # Plot
    if plot:
        if autoclose:
            plt.close('all')
        plt.figure(figsize=figsize)
        plt.plot(th_range, costs)
        plt.tight_layout()
    # Return the threshold
    return opt_th