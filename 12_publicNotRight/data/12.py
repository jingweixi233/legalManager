import json
import jieba
import jieba.posseg as pseg


# 添加词典
jieba.add_word("审理未成年人刑事案件")
jieba.add_word("作案时未满")
jieba.add_word("作案时未成年")
jieba.add_word("犯罪时未满")
jieba.add_word("犯罪时未成年")
jieba.add_word("未满十八岁")
jieba.add_word("未满十八周岁")



f = open("civil.json", 'r', encoding="utf-8")
caseNum = 0
numDivorce = 0
numYoung = 0
flagDivorce = 0
flagYoung = 0
flagRaise = 0
criminal_dic = []
for line in f.readlines():
    flagDivorce = 0
    flagYoung = 0
    flagRaise = 0
    caseNum += 1
    dic = json.loads(line)
    criminal_dic.append(dic)

    if "离婚" in dic["reason"]:
        flagDivorce = 1
    for key in dic.keys():
        criminal_text = dic[key]
        word_list = jieba.cut(criminal_text, cut_all=True)  # 全模式分词
        result = pseg.cut(criminal_text)  # 带词性标注的分词
        # 关键词匹配
        if "审理未成年人刑事案件" in criminal_text \
                or "作案时未满" in criminal_text \
                or "作案时未成年" in criminal_text \
                or "犯罪时未成年" in criminal_text \
                or "犯罪时未满" in criminal_text:
            flagYoung = 1
            
        if "抚养" in criminal_text:
            flagRaise = 1
    if(flagYoung == 1):
        numYoung += 1
        print("（未成年）" + dic["id"] + '\t' + dic["docName"])
    if(flagRaise == 1 and flagDivorce == 1):
        numDivorce += 1
        print("（离婚纠纷）" + dic["id"] + '\t' + dic["docName"])
f.close()

print("案件样本总数为：", caseNum)  # 输出所检测的样本总数
print("裁判文书公开不合法（未成年人）：", numYoung)
print("裁判文书公开不合法（离婚纠纷）：", numDivorce)
