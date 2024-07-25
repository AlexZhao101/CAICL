import numpy as np

def hoi_list():
    dict={}
    with open(r"E:\programming\test_work\ViPLO\hicodet\hico_list_hoi.txt", "r", encoding='utf-8') as f:  #打开文本
        for line in f.readlines():  # 依次读取每行
            line = line.strip().split()  # 去掉每行头尾空白
            dict.setdefault(line[2]+' '+line[1],[]).append(int(line[0])-1)
    return dict

if __name__=='__main__':
    dict=hoi_list()
    
    for key in dict.keys():
        print(key.split(' '))
