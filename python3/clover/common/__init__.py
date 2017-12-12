import os
import shutil

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

def read_csv(fn,col_name_list):
    ret = []
    with open(fn,'r') as fin:
        for line in csv.reader(fin):
            assert(len(line)==len(col_name_list))
            ret.append({col_name_list[i]:line[i] for i in range(len(col_name_list))})
    return ret

def write_csv(fn,v_dict_list,col_name_list,sortkey_func=None):
    if sortkey_func != None:
        v_dict_dict = { sortkey_func(v_dict):v_dict for v_dict in v_dict_list }
        v_dict_dict_key_sort = sorted(v_dict_dict.keys())
        v_dict_list = [ v_dict_dict[k] for k in v_dict_dict_key_sort ]
    with open(fn,'w') as fout:
        csv_out = csv.writer(fout)
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
