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



f = open("dic.json", 'r', encoding="utf-8")

caseNum = 0
caseValid = 0
numDivorce = 0
numYoung = 0
flagDivorce = 0
flagYoung = 0
flagRaise = 0
criminal_dic = []
flagDivorce = 0
flagYoung = 0
flagRaise = 0

dic = json.load(f)
for key in dic.keys():
    caseNum += 1
    caseDivorce = 0
    caseCriminal = 0
    flagDivorce = 0
    flagYoung = 0
    flagVictim = 0
    upNum = 0
    word_dic = []
    if(not("全文" in dic[key].keys())):
        continue
    caseValid += 1
    # 由于所有条文的keys不一致，所以进行全部浏览
    criminal_text = "".join(dic[key]["全文"])
    # 关键词匹配
    if "审理未成年人刑事案件" in criminal_text \
            or "作案时未满" in criminal_text \
            or "作案时未成年" in criminal_text \
            or "犯罪时未成年" in criminal_text \
            or "犯罪时未满" in criminal_text:
        flagYoung = 1

    if("离婚纠纷" in criminal_text):
        flagDivorce = 1
        
    if "抚养" in criminal_text:
        flagRaise = 1
    if(flagYoung == 1):
        numYoung += 1
        print("（未成年）" + key)
    if(flagRaise == 1 and flagDivorce == 1):
        numDivorce += 1
        print("（离婚纠纷）" + key)
f.close()

print("案件样本总数为：", caseNum)  # 输出所检测的样本总数
print("有效案件数: ", caseValid)
print("裁判文书公开不合法（未成年人）：", numYoung)
print("裁判文书公开不合法（离婚纠纷）：", numDivorce)
