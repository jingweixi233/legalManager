# 接口文件

# 输入json文件，处理成我们想要的格式
jsonChange(origin.json){
	
}
input: origin.json
output: criminal.json/civil.json/admin.json


# 将json文件转化为数据list
jsonLoad(){
	
}
input: criminal.json/civil.json/admin.json
output: paper[][]

# 主菜单: 选择需要检测的内容
mainMenu() {

}


# 7.未告知上诉权利
#好像都没有告知，只是有的提起了不服上诉
appealNotKnown()
{
	

}

# 8.匿名未处理
anonymousNotDeal(){
	
}
input: 婚姻家庭、继承纠纷案件中的当事人及其法定代理人；刑事案件被害人及其法定代理人；未成年人及其法定代理人
output: 匿名处理x条，未匿名y条

# 9.隐私信息泄露
privacyNotDeal(){
	

}
input: 自然人的家庭住址，身份证号码，银行账号，车牌号码
output: 隐私处理x条，未匿名y条

# 10.适用法律不正确
legalNotMatch() {

}
input: 所有信息
output: 适用法律不当的docname
	
# 11.案由信息不当
causeNotMatch() {

}
input: docname, reason, 包含"认为"的文本
output: 案由信息不当的docname
	
# 12 文书公开不当
publicNotRight() {

}
input: 所有信息(Criminal), Reason(Civil)
output: 公开不当的docname

# 13 诉请回应缺失
appealNotAnswer() {

}
input: 所有信息
outout: 诉请回应缺失的docname

# 进行总结
printSummary() {

}
