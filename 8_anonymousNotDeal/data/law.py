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

f = open("civil.json", 'r', encoding="utf-8")
num = 0
caseNum = 0
numYoung = 0
numDivorce = 0
numVictim = 0
flagYoung = 0
flagDivorce = 0
criminal_dic = []


for line in f.readlines():
    caseNum += 1
    caseDivorce = 0
    caseCriminal = 0
    flagDivorce = 0
    flagYoung = 0
    flagVictim = 0
    upNum = 0
    dic = json.loads(line)
    criminal_dic.append(dic)
    word_dic = []

    if(dic["caseType"] == "刑事"):
    	caseCriminal = 1
    # 由于所有条文的keys不一致，所以进行全部浏览
    for key in dic.keys():
        criminal_text = dic[key]
        if("离婚纠纷" in criminal_text or "继承纠纷" in criminal_text):
        	caseDivorce = 1
        #word_list = jieba.cut(criminal_text, cut_all=True)  # 全模式分词
        result = pseg.lcut(criminal_text)  # 带词性标注的分词
        # 关键词匹配
        
        for word in result:
        	word_dic.append(str(word));

    lenDic = len(word_dic)
    for i in range(0, lenDic):
    	#案件未成年人处理
    	if("未成年" in word_dic[i]):	
    		for j in range(i - 20, i + 20):
    			if("nr" in word_dic[j] and (not("某" in word_dic[j])) and (not("某" in word_dic[j+1]))):
    				flagYoung = 1;
    	#案件离婚纠纷处理
    	if("当事人" in word_dic[i] or "法定代理人" in word_dic[i]):	
    		
    		if(i + 20 < lenDic):
    			upNum = i + 20
    		else:
    			upNum = lenDic
    		for j in range(i - 20, upNum):
    			if(("nr" in word_dic[j]) and (not("某" in word_dic[j])) and (not("某" in word_dic[j+1]))):
    				flagDivorce = 1;
    	
    	#案件受害人处理
    	if("被害人" in word_dic[i] or "法定代理人" in word_dic[i]):	
    	
    		if(i + 20 < lenDic):
    			upNum = i + 10
    		else:
    			upNum = lenDic
    		if("nr" in word_dic[i+1] and \
    			not(len(word_dic[i+1]) == 4) and \
    			not("某" in word_dic[i+1]) and \
    			not("陈述" in word_dic[i+1])):
    			flagVictim = 1
    if flagYoung == 1:
    	numYoung += 1
    	print ("（未成年）" + dic["id"] + '\t' + dic["docName"])

    if ((flagVictim == 1) and (caseCriminal == 1)):
    	numVictim += 1
    	print ("（被害人）" + dic["id"] + '\t' + dic["docName"])

    if flagDivorce == 1 and caseDivorce == 1:
    	numDivorce += 1
    	print ("（离婚纠纷）" + dic["id"] + '\t' + dic["docName"])


f.close()
print("案件样本总数为：", caseNum)  # 输出所检测的样本总数
print("案件样本公开不当（未成年人）：", numYoung)
print("案件样本公开不当（离婚纠纷）：", numDivorce)
print("案件样本公开不当（被害人）：", numVictim)

