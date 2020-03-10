## 数据说明
### 输入数据

1. criminal.json/civil.json/admin.json:把手网上爬下来的数据
2. word2vec_result.bin：利用HanLP分词后得到split_words.txt，在经过word2vec训练后得到词向量文件

运行时将上述两个文件与py文件放在一起

### 输出格式：result.txt

记录model的准确率，判断错误的案件的真实案由，以及对于每一种案由的预测情况

---
## 代码运行示例
```bash
python Reason.py xx
```
### 参数说明
xx: criminal.json 或 civil.json 或 admin.json

## 实验结果
- 我们的GRU模型
	- dic.json 训练集(352)准确率 96.4%, 测试集(87)准确率 75.3%
	
- Baseline SVM
	- dic.json 训练集(352)准确率 99.7%, 测试集(87)准确率 61.9%

- Baseline Random Forest
	- dic.json 训练集(352)准确率 99.1%, 测试集(87)准确率 69.1%
