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


# 判断被告人年龄
def age_judge(people):
    # 确定审判时间，由于数据的审判年份一致，所以这里简化，直接输入常量
    judge_date = 2014
    seg_list = jieba.cut(people, cut_all=True)
    words_list = []
    for word in seg_list:
        words_list.append(word)
    start = -1
    end = -1
    birth_date = 0
    flag = 0
    # 找到被告人的信息
    for i in range(len(words_list)):
        if words_list[i] == "被告人":
            start = i
            break
    # 找到出生信息
    for j in range(start, len(words_list), 1):
        if words_list[j] == "出生":
            flag = 1
            end = j
            break
        if words_list[j] == "生于":
            if flag == 0:
                flag = 2
            end = j
            break
    # 若flag=0，说明未匹配成功，当作无出生信息记录处理
    # 其他情况则开始锁定出生年份
    if flag == 1:
        for k in range(end, start, -1):
            if words_list[k] == "年":
                if words_list[k-1].isdigit():
                    birth_date = int(words_list[k-1])
                    break
    if flag == 2:
        if words_list[end+2] == "年":
            if words_list[end+1].isdigit():
                birth_date = int(words_list[end+1])
        else:
            for k in range(start, end, 1):
                if words_list[k] == "年":
                    if words_list[k-1].isdigit():
                        birth_date = int(words_list[k - 1])
                        break
    # 计算年龄
    if birth_date > 0:
        age = judge_date - birth_date
        return age
    else:
        return 100


f = open("admin.json", 'r', encoding="utf-8")

caseNum = 0
caseYoung = 0
caseDivorce = 0
flagYoung = 0
flagDivorce = 0
criminal_dic = []

for line in f.readlines():
    caseNum += 1
    flagYoung = 0
    flagDivorce = 0
    dic = json.loads(line)
    criminal_dic.append(dic)
    #未成年人案件一般法案里会说，所以不用判断年龄了
    # 由于所有条文的keys不一致，所以进行全部浏览
    for key in dic.keys():
        criminal_text = dic[key]
        #分词了，但没有用啊？？？
        word_list = jieba.cut(criminal_text, cut_all=True)  # 全模式分词
        result = pseg.cut(criminal_text)  # 带词性标注的分词
        # 关键词匹配
        #案件未成年人处理
        if "审理未成年人刑事案件" in criminal_text \
                or "作案时未满" in criminal_text \
                or "作案时未成年" in criminal_text \
                or "犯罪时未成年" in criminal_text \
                or "犯罪时未满" in criminal_text \
                or "未成年" in criminal_text:
            if (not(key == "相关法律条文" \
            	or key == "附" \
            	or "某" in criminal_text \
            	or "x" in criminal_text \
            	or "未成年人财产" in criminal_text)):
            	flagYoung = 1
            	
        if "离婚" in criminal_text \
        	or "未成年子女抚养" in criminal_text:
        	if (not("某" in criminal_text \
            	or "x" in criminal_text)):
        		flagDivorce = 1
    if flagYoung == 1:
    	caseYoung += 1
    	print (dic["id"] + '\t' + dic["docName"])
    if flagDivorce == 1:
    	caseDivorce += 1
    	print(dic["id"] + '\t' + dic["docName"])

f.close()

print("案件样本总数为：", caseNum)  # 输出所检测的样本总数
print("案件样本公开不当（未成年人）：", caseYoung)
print("案件样本公开不当（离婚纠纷）：", caseDivorce)
