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
numAddress = 0
numBank = 0
numID = 0
flagAddress = 0
flagBank = 0
flagID = 0
criminal_dic = []
criminal_text = ""

dic = json.load(f)
for key in dic.keys():
    caseNum += 1
    flagAddress = 0
    flagBank = 0
    flagID = 0
    upNum = 0
    word_dic = []
    if(not("全文" in dic[key].keys())):
        continue
    caseValid += 1

    # 由于所有条文的keys不一致，所以进行全部浏览
    criminal_text = "".join(dic[key]["全文"])

    #word_list = jieba.cut(criminal_text, cut_all=True)  # 全模式分词
    result = pseg.lcut(criminal_text)  # 带词性标注的分词
    # 关键词匹配
    
    for word in result:
    	word_dic.append(str(word))

    lenDic = len(word_dic)
    for i in range(0, lenDic):
        #案件银行账号处理
        if("银行账号" in word_dic[i]):
            if(i + 20 < lenDic):
                upNum = i + 20
            else:
                upNum = lenDic	
            for j in range(i - 20, upNum):
                if("m" in word_dic[j]):
                    flagBank = 1;
        #案件身份证号处理
        if("身份证号" in word_dic[i]):
            if(i + 20 < lenDic):
                upNum = i + 20
            else:
                upNum = lenDic
            for j in range(i - 20, upNum):
                if(("m" in word_dic[j])):
                    flagID = 1;
    	#案件家庭住址处理
        if("住址" in word_dic[i]):
            if(i + 20 < lenDic):
                upNum = i + 20
            else:
                upNum = lenDic
            for j in range(i - 20, upNum):
                if(("市" in word_dic[j])):
                    flagAddress = 1;
    if flagBank == 1:
    	numBank += 1
    	print ("（银行账号）" + key)

    if (flagID == 1):
    	numID += 1
    	print ("（身份证号）" + key)

    if flagAddress == 1:
    	numAddress += 1
    	print ("（家庭住址）" + key)
    
f.close()
print("案件样本总数为：", caseNum)  # 输出所检测的样本总数
print("有效案件数: ", caseValid)
print("案件信息公开不当（银行账号）：", numBank)
print("案件样本公开不当（身份证号）：", numID)
print("案件样本公开不当（家庭住址）：", numAddress)
