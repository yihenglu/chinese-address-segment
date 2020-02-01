#!/usr/bin/env python 
# -*- coding:utf-8 -*-
'''
根据包含地址的excel文件和地址标签体系产生训练数据
同时对5%的语料删除省与市
版本V3.0：每三十条从中删除“省”、“市”的字眼，
版本V4.0：每三十条选择一条从中删除“**区”
版本V5.0：
1.调整删除“省市区”的概率（即对mod的数进行了调整）
2.加入自定义词典供结巴进行分词
3.解决模型随机性问题
'''
import pandas as pd


train_file = open("../data/cs_nj_184_train5.txt", 'w', encoding='utf8')  # 4表示该数据对应版本4
dev_file = open("../data/cs_nj_184_dev5.txt", 'w', encoding='utf8')
test_file = open("../data/cs_nj_184_test5.txt", 'w', encoding='utf8')

biaoji = ["XZQHS", "XZQHCS", "XZQHQX", "XZQHJD", "XZQHSQ",	"JD1",	"JD2",	"JD3",	"JD4",	"JD5",	"MP1",	"MP2",	"MP3",
   "MP4",	"MP5",	"DYS1",	"DYS2",	"DYS3",	"POI"]

# 读excel
address_file = '../data/view_mpld_dys_cs_nj_184.csv'
# address_file = 'temp_no_use/raw_data.csv'
address_df = pd.read_csv(address_file, sep=',', encoding='utf-8', low_memory=False)  #  dtype={"XZQHS":str, "XZQHCS":str, "XZQHQX":str, "XZQHJD":str, "XZQHSQ":str,	"JD1":str,	"JD2":str,	"JD3":str,	"JD4":str,	"JD5":str,	"MP1":str,	"MP2":str,	"MP3":str,"MP4":str,	"MP5":str,	"DYS1":str,	"DYS2":str,	"DYS3":str,	"POI":str}
print('列名：')
columns_list = list(address_df.columns)
print(columns_list)
# 删除后两列
# address_df.drop('LONGITUDE', axis=1, inplace=True)
# address_df.drop('LATITUDE', axis=1, inplace=True)

# 遍历每一行，做训练集 注意要加空行！！！
for index_line,line in enumerate(address_df.values):
    # print(line)
    for i,item in enumerate(line):  # 遍历每一行中的每一个元素，如：1 成都市
        # print(i, str(item), len(str(item)), str(item) == 'nan')
        # if not np.isnan(item):  # 为Nan的不做处理
        item = str(item).replace('　', '')

        # 如果当前元素是“省”或者“市”，则以5%的概率去除
        if biaoji[i] == "XZQHS" and index_line % 15 == 0:
            item = 'nan'
        if biaoji[i] == "XZQHCS" and index_line % 21 == 0:
            item = 'nan'
        # 每三十条选择一条从中删除“**区”
        if biaoji[i] == "XZQHQX" and index_line % 20 == 0:
            item = 'nan'

        # 每三十条从中删除“省”、“市”的字眼
        if biaoji[i] == "XZQHS" and index_line % 30 == 0:
            item = str(item).replace("省", "")
        if biaoji[i] == "XZQHCS" and index_line % 31 == 0:
            item = str(item).replace("市", "")

        if str(item) != 'nan':
            # print(i,item)
            begin = 0
            for char in item.strip():
                if begin == 0:
                    begin += 1
                    string1 = char + ' ' + 'B-' + biaoji[i] + '\n'
                    if index_line % 10 == 8:  # 划分训练数据
                        dev_file.write(string1)
                    elif index_line %10 == 9:
                        test_file.write(string1)
                    else:
                        train_file.write(string1)
                else:
                    string1 = char + ' ' + 'I-' + biaoji[i] + '\n'
                    # print(string1)
                    if index_line % 10 == 8:  # 划分训练数据
                        dev_file.write(string1)
                    elif index_line %10 == 9:
                        test_file.write(string1)
                    else:
                        train_file.write(string1)
    # 加空行
    if index_line % 10 == 8:  # 划分训练数据
        dev_file.write('\n')
    elif index_line % 10 == 9:
        test_file.write('\n')
    else:
        train_file.write('\n')

print("共{}条样本".format(index_line))

train_file.write('好 O\n')
train_file.close()
dev_file.close()
test_file.close()

