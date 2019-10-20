"""
判断案件是否存在案由不当
作者: 冯伟琪
邮箱: fengweiqi@sjtu.edu.cn
时间: 2019年10月19日
"""
import json
import sys
import word2vec
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
	print(train_data.shape)
	print(test_data.shape)
	# 存入文件 加速开发流程
	np.save("../data/criminal_train_data.npy", train_data)
	np.save("../data/criminal_test_data.npy", test_data)
	np.save("../data/criminal_train_target.npy", train_target)
	np.save("../data/criminal_test_target.npy", test_target)
	np.save("../data/criminal_reason_dict.npy", reason_dict)
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
	early_stopping = EarlyStopping(monitor='val_loss',patience=50,verbose=1,mode='auto')
	train_target = np_utils.to_categorical(train_target, num_classes = reason_num)
	history = model.fit(train_data, train_target, validation_split=0.15,batch_size=500, epochs=500, callbacks=[early_stopping])

	model.save('criminal.h5')

	#可视化训练过程
	plt.figure(1)
	plt.plot(history.history['acc'])
	plt.plot(history.history['val_acc'])
	plt.title('model acc')
	plt.ylabel('acc')
	plt.xlabel('epoch')
	plt.legend(['train','test'],loc='upper left')
	plt.savefig('../data/acc.png')

	plt.figure(2)
	plt.plot(history.history['loss'])
	plt.plot(history.history['val_loss'])
	plt.title('model loss')
	plt.ylabel('loss')
	plt.xlabel('epoch')
	plt.legend(['train','test'],loc='upper right')
	plt.savefig('../data/loss.png')


def get_result(input_train,output_train,input_test,output_test,output_test_c,reasons,reasonNum):
	f=open('result.txt','a')
	model = load_model('model100100.h5')
	result = model.evaluate(input_train,output_train,batch_size=100)
	#print('\nTrain Acc:',result[1])
	f.write('\nTrain Acc:')
	f.write(str(result[1]))
	result = model.evaluate(input_test,output_test_c)
	#print('\nTest Acc:',result[1])
	f.write('\nTest Acc:')
	f.write(str(result[1]))

	print(model.summary())

	#print(np.argmax(model.predict(input_test),axis=1))
	predict = np.argmax(model.predict(input_test),axis=1)

	#打印判断错误的案件的真实案由和输出案由
	count=0
	for j in range(predict.shape[0]):
		if predict[j]!=output_test[j]:
			count=count+1
			#print(j)
			for reason in reasons:
				if reasons[reason]==output_test[j]:
					#print('correct:',reasons[reason],reason)
					correct_reason = reason
				if reasons[reason]==predict[j]:
					#print('predict:',reasons[reason],reason)
					predict_reason = reason
			f.write('\n')
			f.write('correct reason: ')
			f.write(correct_reason)
			f.write(' predict reason: ')
			f.write(predict_reason)
	print('total error:',count,count/predict.shape[0])


	num_of_this_reason_in_test=[0 for i in range(len(reasons))]
	num_of_this_reason_error_in_test=[0 for i in range(len(reasons))]

	#对于每种案由的具体分析
	#以下格式： 案由 数据集中该案由案件数目 测试集中该案由案件数目 测试集中该案由预测错误数目 测试率（测试数目/数据集中总数目） 错误率
	f.write('\n')
	for reason in reasons:
		f.write(reason)
		#f.write('\tnum of cases: ')
		f.write('\t')
		f.write(str(reasonNum[reasons[reason]]))
		for i in range(predict.shape[0]):
			if reasons[reason]==output_test[i]:
				num_of_this_reason_in_test[reasons[reason]]+=1
				if output_test[i]!=predict[i]:
					num_of_this_reason_error_in_test[reasons[reason]]+=1
		#f.write(' total number in test: ')
		f.write('\t')
		f.write(str(num_of_this_reason_in_test[reasons[reason]]))
		#f.write(' error number in test: ')
		f.write('\t')
		f.write(str(num_of_this_reason_error_in_test[reasons[reason]]))
		test_rate=num_of_this_reason_in_test[reasons[reason]]/reasonNum[reasons[reason]]
		#f.write(' test rate: ')
		f.write('\t')
		f.write(str(test_rate))
		if num_of_this_reason_in_test[reasons[reason]]!=0:
			#f.write(' error rate: ')
			f.write('\t')
			f.write(str(num_of_this_reason_error_in_test[reasons[reason]]/num_of_this_reason_in_test[reasons[reason]]))
		f.write('\n')
	f.close()

def split_word():
    #分词用于训练embedding
    f = open('split_words.txt','a')
    for i in range(len(papers)):
	    for word in papers[i]:
		    if word!='id' and word!='caseNo' and word!='judgementDateStart':
			    splits = HanLP.segment(papers[i][word])
			    for split in splits:
				    f.write(split.word)
				    f.write('\t')
			    f.write('\n')


def causeNotMatch():
	"""
	判断案由不当的主函数
	"""
	# 从输入的 json 文件提取案件信息, 并且返回预训练的词向量模型
	# papers, word_embedding = loadData()
	# 从案件信息中提取训练集和测试集的
	# train_data, train_target, test_data, test_target, reason_dict = preprocessData(papers, word_embedding)
	train_data = np.load("../data/criminal_train_data.npy")
	test_data = np.load("../data/criminal_test_data.npy")
	train_target = np.load("../data/criminal_train_target.npy")
	test_target = np.load("../data/criminal_test_target.npy")
	reason_dict = np.load("../data/criminal_reason_dict.npy", allow_pickle=True).item()
	trainModel(train_data, train_target, len(reason_dict))
	# get_result(input_train,output_train,input_test,output_test,output_test_c,reasons,reasonNum)
	print('success!!')

if __name__=='__main__':
	causeNotMatch()


# 案由个数
# criminal 137
# civil 180
# admin 36
'''
id
docName		*
court
caseNo
caseType
instrumentType
reason
procedureId
referenceType
judgementDateStart
当事人
审理经过
公诉机关称
本院查明
本院认为	*
裁判结果
审判人员
'''
