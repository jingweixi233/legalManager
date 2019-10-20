"""
判断案件是否存在案由不当
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
		papers.append(dic)
	# 导入预训练的 word2vec model
	word_embedding = word2vec.load('../data/word2vec_result.bin')
	print("---已经成功导入数据 {}---".format(filename))

	return papers, word_embedding


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
	word_nums = 100
	# "认为" 中提取的关键词数量
	word_num_1 = 97
	# "docName" 中提取的关键词数量
	word_num_2 = word_nums - word_num_1
	# 创建 "reason" 字典 ("reason": "reason id")
	reasons = list(set([paper["reason"] for paper in papers]))
	reason_pairs = [(reason, reason_id) for (reason_id, reason) in enumerate(reasons)]
	reason_dict = dict(reason_pairs)
	# 依次对每一个案件进行处理
	for (paper_id, paper) in enumerate(papers):
		# 提取含有认为信息的所有 keys
		keys = [item for item in paper if "认为" in item]
		# 判断当前案件是否有效
		if len(keys) == 0:
			continue
		# 提取有效案件信息
		key = keys[0]
		length = len(HanLP.segment(paper[key]))
		# 从 "认为" 对应的信息中提取 
		words = list(HanLP.extractKeyword(paper[key], min(word_num_1, length)))
		# 从 "docname" 对应的信息进行提取
		words += list(HanLP.extractKeyword(paper['docName'], word_num_2))
		# 去除无用的字符
		if_valid = lambda word: word != '\n' and word != ' '
		words = [word for word in words if if_valid(word)]
		# 得到关键词编码的词向量
		vecs = [word_embedding[word].reshape(1, -1) for word in words]
		# 将其处理成矩阵
		word_matrix = np.concatenate(vecs, 0)	
		# 填充均值使其为 (1, 100, 100)的形状
		pad_width = ((0, word_nums - word_matrix.shape[0]), (0, 0))
		word_matrix = np.pad(word_matrix, pad_width, 'mean')
		word_matrix = np.expand_dims(word_matrix, 0)
		# 提取原因
		reason = paper['reason']
		# 根据随机数分到 训练集 或者是 测试集中
		if np.random.rand() < train_data_ratio:
			train_data.append(word_matrix)
			train_target.append(reason_dict[reason])
		else:
			test_data.append(word_matrix)
			test_target.append(reason_dict[reason])

	# 整理成 numpy 3D array 格式
	train_data = np.concatenate(train_data)
	test_data = np.concatenate(test_data)
	train_target = np.array(train_target)
	test_target = np.array(test_target)
	# 存入文件 加速开发流程
	np.save("../data/civil/civil_train_data.npy", train_data)
	np.save("../data/civil/civil_test_data.npy", test_data)
	np.save("../data/civil/civil_train_target.npy", train_target)
	np.save("../data/civil/civil_test_target.npy", test_target)
	np.save("../data/civil/civil_reason_dict.npy", reason_dict)
	print("---已经成功产生训练集和测试集---")
	return train_data, train_target, test_data, test_target, reason_dict


def trainModel(train_data, train_target, reason_num):
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
	model.add(GRU(384,input_shape=(100,100)))
	model.add(Dropout(0.5))
	model.add(Dense(units=384,activation = 'relu'))
	model.add(Dropout(0.5))
	model.add(Dense(units=384,kernel_regularizer=regularizers.l2(0.01),activation = 'relu'))
	model.add(Dropout(0.5))
	model.add(Dense(units=384,activation = 'relu'))
	model.add(Dropout(0.5))
	model.add(Dense(units=384,activation = 'relu'))
	model.add(Dropout(0.5))
	model.add(Dense(units=reason_num, activation = 'softmax'))

	# 编译神经网络模型
	model.compile(loss='categorical_crossentropy',optimizer = 'adam',metrics=['accuracy'])
	# 防止过拟合
	train_target = np_utils.to_categorical(train_target, num_classes=reason_num)
	# 开始训练模型
	history = model.fit(train_data, train_target, validation_split=0.15, batch_size=500, epochs=500)
	model.save('../model/civil.h5')
	print("---训练模型完毕---")

def testModel(train_data, train_target, test_data, test_target, reason_num):
	# 加载预训练模型
	model = load_model("../model/civil.h5")
	# 转化成 one-hot 矩阵
	train_target = np_utils.to_categorical(train_target, num_classes=reason_num)
	# 转化成 one-hot 矩阵
	test_target = np_utils.to_categorical(test_target, num_classes=reason_num)
	result = model.evaluate(train_data, train_target)
	print("训练集准确率: {:.3f}".format(result[1]))
	result = model.evaluate(test_data, test_target)
	print("测试集准确率: {:.3f}".format(result[1]))


def causeNotMatch():
	"""
	判断案由不当的主函数
	"""
	# 不显示无用信息
	os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
	# 从输入的 json 文件提取案件信息, 并且返回预训练的词向量模型
	papers, word_embedding = loadData()
	# 从案件信息中提取训练集和测试集的
	train_data, train_target, test_data, test_target, reason_dict = preprocessData(papers, word_embedding)
	# 训练我们的模型
	trainModel(train_data, train_target, len(reason_dict))
	# 测试我们的模型
	testModel(train_data, train_target, test_data, test_target, len(reason_dict))
	print("---案由不当检测完毕---")

	
if __name__=='__main__':
	causeNotMatch()

