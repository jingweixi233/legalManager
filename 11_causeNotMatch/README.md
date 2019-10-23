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
	- criminal.json 训练集(3567)准确率 87.4%, 测试集(930)准确率 72.5%
	- civil.json	训练集(1273)准确率 86.9%, 测试集(317)准确率 69.1%
	- admin.json	训练集(232) 准确率 89.2%, 测试集(63 )准确率 47.6%
	
- Baseline SVM
	- criminal.json 训练集(3567)准确率 51.0%, 测试集(930)准确率 38.1%
	- civil.json	训练集(1273)准确率 37.1%, 测试集(317)准确率 33.1%
	- admin.json	训练集(232) 准确率 21.1%, 测试集(63 )准确率 15.8%

- Baseline Random Forest
	- criminal.json 训练集(3567)准确率 33.3%, 测试集(930)准确率 29.1%
	- civil.json	训练集(1273)准确率 35.1%, 测试集(317)准确率 30.2%
	- admin.json	训练集(232) 准确率 40.5%, 测试集(63 )准确率 23.8%
