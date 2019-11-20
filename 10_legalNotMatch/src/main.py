"""
判断案件是否存在适用法律不当
作者: 冯伟琪
邮箱: fengweiqi@sjtu.edu.cn
时间: 2019年10月19日
"""
import json
import sys
import word2vec
import os
import matplotlib.pyplot as plt
from pyhanlp import *
import numpy as np
from keras.models import Sequential
from keras.layers.core import Dense,Dropout,Activation
from keras.layers import LSTM,GRU,ConvLSTM2D,Conv2D,MaxPooling2D,Flatten
from keras.optimizers import SGD,Adam
from keras.utils import np_utils
from keras.models import load_model
from keras import regularizers
from keras.callbacks import EarlyStopping
import re
# Import for testing baseline
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier

INPUT = 30

def loadData():
	"""
	导入json格式的案件数据

	参数:
		None
	
	返回:
		papers: 		含有案件信息的list
		word_embedding: 预先训练的 word2vec 模型
	"""
	papers = []
	# 第一个参数作为被读入的json文件路径
	filename = sys.argv[1]
	file = open(filename,'r')
	# 将读取的数据存入 papers list 中
	for line in file.readlines():
		dic = json.loads(line)

	for key in dic.keys():
		papers.append(dic[key])
	# 导入预训练的 word2vec model
	word_embedding = word2vec.load('../data/word2vec_result.bin')
	print("---已经成功导入数据 {}---".format(filename))

	return papers, word_embedding

def extractLaw(law):
	# 只保留第一条法律
	begin = -1
	end = -1
	for i in range(len(law)):
		if begin == -1 and law[i] == "《":
			begin = i
			break
	for i in range(begin, len(law)):
		if law[i] == "，" or law[i] == "、":
			end = i
			break
	if end != -1:
		law = law[begin: end]
	else:
		law = law[begin:]

	law = re.findall(r"《.*》第.*条", law)
	return law

def preprocessData(papers, word_embedding):
	"""
	对案件关键词编码, 并生成训练集和测试集

	参数:
		papers: 		含有案件信息的list
		word_embedding: 预先训练好的word2vec 模型
	
	返回:
		train_data: 	训练集数据
		train_target: 	训练集标签
		test_data:		测试集数据
		test_label:		测试集标签	
	"""
	# 每一个案件所编码得到的词矩阵为 (1, 100, 100)
	# 一行中有100个特征，即一个词用100维度向量表示
	# 一共100行，即100个词语
	train_data = []
	train_target = []
	test_data = []
	test_target = []
	train_data_ratio = 0.8
	# 每一个案件抽取的关键词数量
	word_nums = INPUT
	# "认为" 中提取的关键词数量
	word_num_1 = 20
	# "全文" 中提取的关键词数量
	word_num_2 = word_nums - word_num_1
	# 创建 "法律条文" 字典 ("法律条文": "法律条文 ID")

	laws = [paper["法律条文名称"][0] for paper in papers if '法律条文名称' in paper.keys() and paper['法律条文名称'] != []]
	allLaws = []
	# 提取法律条文信息
	for law in laws:
		law = extractLaw(law)
		if law != []:
			allLaws.append(law[0])
	laws = list(set(allLaws))
	law_pairs = [(law, law_id) for (law_id, law) in enumerate(laws)]
	law_dict = dict(law_pairs)

	# 依次对每一个案件进行处理
	for (paper_id, paper) in enumerate(papers):
		if '本院认为' not in paper.keys() or paper['本院认为'] == [] or '法律条文名称' not in paper.keys() or paper['法律条文名称'] == []:
			continue
		# 提取含有认为信息的所有 keys
		keys = [item for item in paper if "认为" in item]
		# 判断当前案件是否有效
		if len(keys) == 0:
			continue
		# 提取有效案件信息
		key = keys[0]
		length = len(HanLP.segment(paper[key][0]))
		# 从 "认为" 对应的信息中提取 
		words = list(HanLP.extractKeyword(paper[key][0], min(word_num_1, length)))
		# 从 "docname" 对应的信息进行提取
		length = len(HanLP.segment(paper['全文'][0]))
		words += list(HanLP.extractKeyword(paper['全文'][0], min(word_num_2, length)))
		# 去除无用的字符
		if_valid = lambda word: word != '\n' and word != ' '
		words = [word for word in words if if_valid(word)]
		# 得到关键词编码的词向量
		vecs = [word_embedding[word].reshape(1, -1) for word in words if word in word_embedding.vocab]
		print(len(vecs))
		# 将其处理成矩阵
		word_matrix = np.concatenate(vecs, 0)	
		# 填充均值使其为 (1, 30, 100)的形状
		pad_width = ((0, word_nums - word_matrix.shape[0]), (0, 0))
		word_matrix = np.pad(word_matrix, pad_width, 'mean')
		word_matrix = np.expand_dims(word_matrix, 0)
		# 提取原因
		law = extractLaw(paper['法律条文名称'][0])
		if law == []:
			continue
		# 根据随机数分到 训练集 或者是 测试集中
		if np.random.rand() < train_data_ratio:
			train_data.append(word_matrix)
			train_target.append(law_dict[law[0]])
		else:
			test_data.append(word_matrix)
			test_target.append(law_dict[law[0]])
		print("处理进度: {} / {}".format(paper_id, len(papers)))
	# 整理成 numpy 3D array 格式
	train_data = np.concatenate(train_data)
	test_data = np.concatenate(test_data)
	train_target = np.array(train_target)
	test_target = np.array(test_target)
	# 存入文件 加速开发流程
	print("---已经成功产生训练集和测试集---")
	np.save("../data/dic/dic_train_data.npy", train_data)
	np.save("../data/dic/dic_train_target.npy", train_target)
	np.save("../data/dic/dic_test_data.npy", test_data)
	np.save("../data/dic/dic_test_target.npy", test_target)
	np.save("../data/dic/dic_law_dict.npy", law_dict)
	return train_data, train_target, test_data, test_target, law_dict


def trainModel(train_data, train_target, law_num):
	"""
	训练神经网络模型, 并保存模型训练文件

	参数:
		train_data: 	训练数据
		train_target:	训练标签
		reason_num:		分类案由的数目

	返回: 
		None
	"""
	# 定义神经网络模型
	model = Sequential()
	model.add(GRU(384,input_shape=(INPUT,100)))
	model.add(Dropout(0.5))
	model.add(Dense(units=384,activation = 'relu'))
	model.add(Dropout(0.5))
	model.add(Dense(units=384,kernel_regularizer=regularizers.l2(0.01),activation = 'relu'))
	model.add(Dropout(0.5))
	model.add(Dense(units=384,activation = 'relu'))
	model.add(Dropout(0.5))
	model.add(Dense(units=384,activation = 'relu'))
	model.add(Dropout(0.5))
	model.add(Dense(units=law_num, activation = 'softmax'))

	# 编译神经网络模型
	model.compile(loss='categorical_crossentropy',optimizer = 'adam',metrics=['acc'])
	# 防止过拟合
	train_target = np_utils.to_categorical(train_target, num_classes=law_num)
	# 开始训练模型
	history = model.fit(train_data, train_target, validation_split=0.15, batch_size=50, epochs=1500)
	saveFig(history, "../visual/dic_loss.png", "../visual/dic.png")
	model.save('../model/dic.h5')
	print("---训练模型完毕---")


def saveFig(history, loss_file, acc_file):
	"""
	可视化并保存图片

	参数:
		history: 	keras model.fit() 的返回值
		loss_file:	loss 可视化图片名
		acc_file:	acc 可视化图片名

	返回:	
		None
	"""
	# Plot training & validation accuracy values
	plt.plot(history.history['acc'])
	plt.plot(history.history['val_acc'])
	plt.title('Model accuracy')
	plt.ylabel('Accuracy')
	plt.xlabel('Epoch')
	plt.legend(['Train', 'Test'], loc='upper left')
	plt.savefig(loss_file)
	plt.clf()
	# Plot training & validation loss values
	plt.plot(history.history['loss'])
	plt.plot(history.history['val_loss'])
	plt.title('Model loss')
	plt.ylabel('Loss')
	plt.xlabel('Epoch')
	plt.legend(['Train', 'Test'], loc='upper left')
	plt.savefig(acc_file)

def testModel(train_data, train_target, test_data, test_target, law_num):
	"""
	将训练好的模型在训练集和测试集进行测试, 并打印相关信息

	参数:
		train_data: 	训练数据
		trian_target:	训练标签
		test_data:		测试数据
		test_target:	测试标签
	
	返回:
		None
	"""
	# 加载预训练模型
	model = load_model("../model/dic.h5")
	# 转化成 one-hot 矩阵
	train_target = np_utils.to_categorical(train_target, num_classes=law_num)
	# 转化成 one-hot 矩阵
	test_target = np_utils.to_categorical(test_target, num_classes=law_num)
	result = model.evaluate(train_data, train_target)
	print("训练集准确率: {:.3f}".format(result[1]))
	result = model.evaluate(test_data, test_target)
	print("测试集准确率: {:.3f}".format(result[1]))


def legalNotMatch():
	"""
	判断案由不当的主函数
	"""
	# 不显示无用信息
	os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
	# 从输入的 json 文件提取案件信息, 并且返回预训练的词向量模型
	papers, word_embedding = loadData()
	# 从案件信息中提取训练集和测试集的
	# train_data, train_target, test_data, test_target, law_dict = preprocessData(papers, word_embedding)

	train_data, train_target, test_data, test_target, law_dict = loadDicData()
	# 训练我们的模型
	trainModel(train_data, train_target, len(law_dict))
	# 测试我们的模型
	testModel(train_data, train_target, test_data, test_target, len(law_dict))

	print("---案由不当检测完毕---")

def loadDicData():
	"""
	导入实现处理好的 Dic 数据
	"""
	train_data = np.load("../data/dic/dic_train_data.npy")
	train_target = np.load("../data/dic/dic_train_target.npy")
	test_data = np.load("../data/dic/dic_test_data.npy")
	test_target = np.load("../data/dic/dic_test_target.npy")
	reason_dict = np.load("../data/dic/dic_law_dict.npy", allow_pickle=True).item()

	return train_data, train_target, test_data, test_target, reason_dict


def baselineSVM():
	"""
	用 SVM 作为 baseline 模型
	"""
	# 导入 Criminal 的数据
	train_data, train_target, test_data, test_target, law_dict = loadDicData()
	
	# 重新 Reshape 训练数据和测试数据
	train_data = train_data.reshape(train_data.shape[0], -1)
	test_data = test_data.reshape(test_data.shape[0], -1)
	train_target = train_target.reshape(-1, 1)
	test_target = test_target.reshape(-1, 1)
	# 用 SVM 作为baseline
	clf = SVC(gamma=10.0, verbose=True)
	clf.fit(train_data, train_target)
	print(clf.score(train_data, train_target))
	print(clf.score(test_data, test_target))


def baselineRandomForest():
	"""
	用 Random Forest 作为 baseline
	"""
	# 导入 Criminal 的数据
	train_data, train_target, test_data, test_target, law_dict = loadDicData()
	train_data = train_data.reshape(train_data.shape[0], -1)
	test_data = test_data.reshape(test_data.shape[0], -1)
	train_target = train_target.reshape(-1, 1)
	test_target = test_target.reshape(-1, 1)
	# 用 SVM 作为baseline
	clf = RandomForestClassifier(n_estimators=100, max_depth=20, random_state=0)
	clf.fit(train_data, train_target)
	print(clf.score(train_data, train_target))
	print(clf.score(test_data, test_target))


if __name__=='__main__':
	baselineRandomForest()


