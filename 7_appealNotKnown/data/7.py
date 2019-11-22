#!/usr/local/bin/python
# -*- coding: utf-8 -*-

import json
import jieba
import jieba.posseg as pseg


'''
word_file = open("LegalwordDict.txt", 'r', encoding="utf-8")
words = word_file.read()
word_list = words.split('\n')
print(word_list)
'''
# 添加词典
#jieba.load_userdict("LegalwordDict.txt")
jieba.add_word("审理未成年人刑事案件")
jieba.add_word("作案时未满")
jieba.add_word("作案时未成年")
jieba.add_word("犯罪时未满")
jieba.add_word("犯罪时未成年")
jieba.add_word("未满十八岁")
jieba.add_word("未满十八周岁")
jieba.add_word("未成年")


#并行分词
jieba.enable_parallel(4)

f = open("dic.json", 'r', encoding="utf-8")
num = 0
caseNum = 0
caseValid = 0
numYoung = 0
numDivorce = 0
numVictim = 0
flagYoung = 0
flagDivorce = 0
criminal_dic = []
criminal_text = ""
caseTold = 0
caseFirst = 0
caseOther = 0
dic = json.load(f)
for key in dic.keys():
    caseNum += 1
    caseDivorce = 0

    flagDivorce = 0
    flagYoung = 0
    flagVictim = 0
    upNum = 0
    word_dic = []
    if(not("全文" in dic[key].keys())):
        continue
    caseValid += 1
    if("一审" in dic[key]["审理类型"]):
    	caseFirst += 1

    if("是否告诉上诉权利"in dic[key].keys()):
    	
    	if(not(dic[key]["是否告诉上诉权利"] == [])):
    		
    		if("一审" in dic[key]["审理类型"]):
    			caseTold += 1

f.close()
print("案件样本总数为：", caseNum)  # 输出所检测的样本总数
print("一审案件数: ", caseFirst)
print("告诉上诉权利的案件数目: ", caseTold)



