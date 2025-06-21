# -*- coding: utf-8 -*-
from django.http import JsonResponse
from django.shortcuts import render
from util.pre_load import neo4jconn, course_dict

import json
import logging


# 创建日志记录器
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

# 添加控制台处理器
ch = logging.StreamHandler()
ch.setFormatter(formatter)
logger.addHandler(ch)
#学科知识点查询
def search_course_knowledgepoint(request):
	ctx = {}
	# #根据传入的实体名称搜索出关系
	print(f"收到学科知识点查询请求")  # 打印返回数据
	if(request.GET):
		entity = request.GET['user_text']
		print(entity)
		if entity in course_dict.keys():
			entity = course_dict.get(entity)
		print(entity)
		entityRelation = neo4jconn.course_knowledgepoint(entity)
		# print(f"返回数据: {entityRelation}")  # 打印返回数据
		if entity=="C language course":
			return render(request, 'course.html', {'entityRelation': json.dumps(entityRelation, ensure_ascii=False)})
		else:
			# if len(entityRelation) == 0:
			# 若数据库中无法找到该实体，则返回数据库中无该实体
			ctx = {'title': '<h2>知识库中暂未添加该实体</h1>'}
			return render(request, 'course.html', {'ctx': json.dumps(ctx, ensure_ascii=False)})

	#需要进行类型转换
	return render(request, 'course.html', {'ctx':ctx})
#知识图谱演变
def course_knowledgepoint_page(request):
    # 返回静态页面，用于展示图谱和输入框
    return render(request, 'course2.html')
def add_course_knowledgepoint(request):
	# #根据传入的实体名称搜索出关系
	if request.method == 'GET':
		# ---------- 处理前端添加三元组请求 ----------
		print("GET参数:", request.GET)
		try:
			source = request.GET.get('source')
			relation = request.GET.get('relation')
			target = request.GET.get('target')
			print(f"收到三元组添加请求: {source} -[{relation}]-> {target}")
			# 添加三元组到 Neo4j 数据库
			neo4jconn.create_relation(source, relation, target)
		except Exception as e:
			print(f"添加三元组出错: {e}")
		entity = "C language course"
		entityRelation = neo4jconn.course_knowledgepoint(entity)
		# print(f"返回数据: {entityRelation}")  # 打印返回数据
		if entity=="C language course":
			return JsonResponse({
				"status": "success",
				"message": "三元组添加成功",
				"entityRelation": entityRelation
			}, json_dumps_params={'ensure_ascii': False})
		else:
			return JsonResponse({
				"status": "error",
				"message": f"添加失败: {str(e)}"
			})
	#需要进行类型转换
	return JsonResponse({
		"status": "error",
		"message": "仅支持 GET 请求"
	})

#题目知识点查询
def search_question_knowledgepoint(request):
	ctx = {}
	# 根据传入的实体名称搜索出关系
	if (request.GET):
		entity = request.GET['user_text']
		entityRelation = neo4jconn.question_knowledgepoint(entity)
		if len(entityRelation) == 0:
			# 若数据库中无法找到该实体，则返回数据库中无该实体
			ctx = {'title': '<h2>知识库中暂未添加该实体</h1>'}
			return render(request, 'question.html', {'ctx': json.dumps(ctx, ensure_ascii=False)})
		else:
			return render(request, 'question.html', {'entityRelation': json.dumps(entityRelation, ensure_ascii=False)})
	# 需要进行类型转换
	return render(request, 'question.html', {'ctx': ctx})

#知识点关系查询
def search_relation(request):
	ctx = {}
	if(request.GET):
		# 实体1
		entity1 = request.GET['entity1_text']
		# 关系
		relation = request.GET['relation_name_text']
		# 实体2
		entity2 = request.GET['entity2_text']
		# 将关系名转为大写
		relation = relation.upper()

		if entity1 in course_dict.keys():
			entity1 = course_dict.get(entity1)
		if entity2 in course_dict.keys():
			entity2 = course_dict.get(entity2)

		# 保存返回结果
		searchResult = {}

		# 1.若只输入entity1,则输出与entity1有直接关系的实体和关系
		if(len(entity1) != 0 and len(relation) == 0 and len(entity2) == 0):
			searchResult = neo4jconn.findRelationByEntity1(entity1)
			if(len(searchResult)>0):
				return render(request,'relation.html',{'searchResult':json.dumps(searchResult,ensure_ascii=False)})

		# 2.若只输入entity2则,则输出与entity2有直接关系的实体和关系
		if(len(entity2) != 0 and len(relation) == 0 and len(entity1) == 0):
			searchResult = neo4jconn.findRelationByEntity2(entity2)
			if(len(searchResult)>0):
				return render(request,'relation.html',{'searchResult':json.dumps(searchResult,ensure_ascii=False)})

		# 3.若输入entity1和relation，则输出与entity1具有relation关系的其他实体
		if(len(entity1)!=0 and len(relation)!=0 and len(entity2) == 0):
			searchResult = neo4jconn.findOtherEntities(entity1,relation)
			if(len(searchResult)>0):
				return render(request,'relation.html',{'searchResult':json.dumps(searchResult,ensure_ascii=False)})

		# 4.若输入entity2和relation，则输出与entity2具有relation关系的其他实体
		if(len(entity2)!=0 and len(relation)!=0 and len(entity1) == 0):
			searchResult = neo4jconn.findOtherEntities2(entity2,relation)
			if(len(searchResult)>0):
				return render(request,'relation.html',{'searchResult':json.dumps(searchResult,ensure_ascii=False)})

		# 5.若输入entity1和entity2,则输出entity1和entity2之间的关系
		if(len(entity1) !=0 and len(relation) == 0 and len(entity2)!=0):
			searchResult = neo4jconn.findRelationByEntities(entity1,entity2)
			if(len(searchResult)>0):
				return render(request,'relation.html',{'searchResult':json.dumps(searchResult,ensure_ascii=False)})

		# 6.若输入entity1,entity2和relation,则输出entity1、entity2是否具有相应的关系
		if(len(entity1)!=0 and len(entity2)!=0 and len(relation)!=0):
			print(relation)
			searchResult = neo4jconn.findEntityRelation(entity1,relation,entity2)
			if(len(searchResult)>0):
				return render(request,'relation.html',{'searchResult':json.dumps(searchResult,ensure_ascii=False)})

		# 7.若全为空
		if(len(entity1)!=0 and len(relation)!=0 and len(entity2)!=0 ):
			pass

		ctx= {'title' : '<h1>暂未找到相应的匹配</h1>'}
		return render(request,'relation.html',{'ctx':ctx})

	return render(request,'relation.html',{'ctx':ctx})
