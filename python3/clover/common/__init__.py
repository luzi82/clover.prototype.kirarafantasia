import os
import shutil
import csv
import json

PHI = (1+5**0.5)/2

def between(a,b,c):
    return (a<=b) and (b<c)

def in_rect(pos,rect):
    return between(rect[0],pos[0],rect[2]) and between(rect[1],pos[1],rect[3])

def makedirs(path):
    if not os.path.isdir(path):
        os.makedirs(path)

def reset_dir(out_dir):
    shutil.rmtree(out_dir,ignore_errors=True)
    os.makedirs(out_dir)

def read_csv(fn):
    col_name_list = None
    ret = []
    with open(fn,'r') as fin:
        for line in csv.reader(fin):
            if col_name_list is None:
                col_name_list = list(line)
            else:
                assert(len(line)==len(col_name_list))
                ret.append({col_name_list[i]:line[i] for i in range(len(col_name_list))})
    return ret

def write_csv(fn,v_dict_list,col_name_list=None,sort_key=None):
    if col_name_list is None:
        assert(len(v_dict_list)>0)
        col_name_list = list(sorted(v_dict_list[0].keys()))
    if sort_key is not None:
        v_dict_dict = { v_dict[sort_key]:v_dict for v_dict in v_dict_list }
        v_dict_dict_key_sort = sorted(v_dict_dict.keys())
        v_dict_list = [ v_dict_dict[k] for k in v_dict_dict_key_sort ]
    with open(fn,'w') as fout:
        csv_out = csv.writer(fout)
        csv_out.writerow(col_name_list)
        for v_dict in v_dict_list:
            csv_out.writerow([v_dict[col_name] for col_name in col_name_list])

def readlines(fn):
    with open(fn,'rt',encoding='utf-8') as fin:
        ret = fin.readlines()
    ret = [ i.strip() for i in ret ]
    return ret

def writelines(fn, txt_list):
    with open(fn, mode='wt', encoding='utf-8') as fout:
        for txt in txt_list:
            fout.write('{}\n'.format(txt))

def appendlines(fn, txt_list):
    with open(fn, mode='at', encoding='utf-8') as fout:
        for txt in txt_list:
            fout.write('{}\n'.format(txt))

def read_json(fn):
    if not os.path.isfile(fn):
        return None
    with open(fn,'r') as fin:
        return json.load(fin)

def write_json(fn,j):
    with open(fn, 'w') as fout:
        json.dump(j, fp=fout, indent=2, sort_keys=True)
        fout.write('\n')
