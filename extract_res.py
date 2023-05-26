import os
import json

keys = ['s_nce', 'length', 'grad_adv', 'hard']

def get_acc_agg_lp(lines, dict):
    acc = eval(lines[-2].strip().split(' ')[-1])
    lp = eval(lines[-1].strip().split('/t')[-1].split(' ')[-1])
    agg = eval(lines[-1].strip().split('/t')[0].split(' ')[-1])
    dict['acc'].append(acc)
    dict['agg'].append(agg)
    dict['lp'].append(lp)
    return dict

def get_other(line, dict):
    line = line.strip()
    if line.split(' ')[-2][:-1] in keys:
        dict[line.split(' ')[-2][:-1]].append(eval(line.split(' ')[-1]))
    
    return dict
    
def split_files(root):
    files = os.listdir(root)
    file_list = []
    for f in files:
        if f.startswith('log') and f.split('_')[-1].startswith('0'):
            file_list.append(os.path.join(root, f))
    return file_list

if __name__ == "__main__":
    root = '/data1/lijingru/DFKD/checkpoints/datafree-adadfkd/'
    lists = split_files(root)
    dict = {
        'teacher':[],
        'student':[],
        's_nce':[],
        'hard':[],
        'length':[],
        'grad_adv':[],
        'acc':[],
        'agg':[],
        'lp':[],
    }
    for f in lists:
        print(f)
        path = f.split('/')[-1]
        dict['teacher'].append(path.split('-')[2])
        dict['student'].append(path.split('-')[3])
        lines = open(f, 'r').readlines()
        for line in lines:
            if line.split(' ')[3] == 'INFO:':
                dict = get_other(line, dict)

        dict = get_acc_agg_lp(lines, dict)

    

    with open('output_res.json', 'w') as f:
        res = json.dumps(dict)
        f.write(res)
