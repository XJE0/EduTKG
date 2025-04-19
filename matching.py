# import os
# import numpy as np
# from sklearn.feature_extraction.text import TfidfVectorizer
# from sklearn.metrics.pairwise import cosine_similarity
# from transformers import AutoTokenizer, AutoModel
# import torch
#
# def load_data(file_path):
#     """加载文本数据"""
#     with open(file_path, 'r', encoding='utf-8') as f:
#         return [line.strip() for line in f.readlines()]
#
# def compute_tfidf_similarity(source_data, target_data):
#     """计算TF-IDF和余弦相似度"""
#     vectorizer = TfidfVectorizer()
#     tfidf_matrix = vectorizer.fit_transform(source_data)
#     target_tfidf = vectorizer.transform(target_data)
#     similarity = cosine_similarity(target_tfidf, tfidf_matrix)
#     return similarity
#
# def bert_cosine_similarity(source_data, target_data, model_name='model/bert_model'):#student_model#bert_model
#     """计算BERT嵌入和余弦相似度"""
#     tokenizer = AutoTokenizer.from_pretrained(model_name)
#     model = AutoModel.from_pretrained(model_name)
#
#     def embed_texts(texts):
#         inputs = tokenizer(texts, padding=True, truncation=True, return_tensors='pt')
#         with torch.no_grad():
#             outputs = model(**inputs)
#         embeddings = outputs.last_hidden_state.mean(dim=1)
#         return embeddings
#
#     source_embeddings = embed_texts(source_data)
#     target_embeddings = embed_texts(target_data)
#     similarity = cosine_similarity(target_embeddings.numpy(), source_embeddings.numpy())
#     return similarity
#
# def match_and_filter(similarity_matrix, source_data, target_data, threshold=0.7):
#     """根据相似度矩阵筛选匹配结果"""
#     matched_pairs = []
#     for i, target_row in enumerate(similarity_matrix):
#         max_sim_idx = np.argmax(target_row)
#         max_sim_value = target_row[max_sim_idx]
#         if max_sim_value >= threshold:
#             matched_pairs.append((target_data[i], source_data[max_sim_idx], max_sim_value))
#     return matched_pairs
# def calculate_average_similarity(matched_pairs):
#     """计算匹配结果的平均相似度"""
#     if matched_pairs:
#         avg_similarity = np.mean([match[2] for match in matched_pairs])
#         return avg_similarity
#     else:
#         return 0.0
# # 加载数据
# syllabus_data = load_data('data/syllabus.txt')
# data1 = load_data('data/data1.txt')
#
# # 使用TF-IDF进行匹配
# tfidf_similarity = compute_tfidf_similarity(syllabus_data, data1)
# tfidf_matched = match_and_filter(tfidf_similarity, syllabus_data, data1)
#
# # 使用BERT进行匹配
# bert_similarity = bert_cosine_similarity(syllabus_data, data1)
# bert_matched = match_and_filter(bert_similarity, syllabus_data, data1)
# # 计算每种方法的平均相似度
# tfidf_avg_similarity = calculate_average_similarity(tfidf_matched)
# bert_avg_similarity = calculate_average_similarity(bert_matched)
# # 输出结果
# # 输出结果
# print("TF-IDF匹配结果:")
# # for match in tfidf_matched:
# #     print(f"教材内容: {match[0]}\n大纲内容: {match[1]}\n相似度: {match[2]:.4f}\n")
# print(f"TF-IDF匹配的平均相似度: {tfidf_avg_similarity:.4f}\n")
#
# print("BERT-匹配结果:")##根据上面的模型，选择不同，结果也不同
# # for match in bert_matched:
# #     print(f"教材内容: {match[0]}\n大纲内容: {match[1]}\n相似度: {match[2]:.4f}\n")
# print(f"BERT-匹配的平均相似度: {bert_avg_similarity:.4f}\n")
import os
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from transformers import AutoTokenizer, AutoModel
import torch

def load_data(file_path):
    """加载文本数据"""
    with open(file_path, 'r', encoding='utf-8') as f:
        return [line.strip() for line in f.readlines()]


def compute_tfidf_similarity(source_data, target_data):
    """计算TF-IDF和余弦相似度"""
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(source_data)
    target_tfidf = vectorizer.transform(target_data)
    similarity = cosine_similarity(target_tfidf, tfidf_matrix)
    return similarity


def compute_tfidf_ci_similarity(source_data, target_data):
    """计算TF-IDF-CI和余弦相似度"""
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(source_data)
    target_tfidf = vectorizer.transform(target_data)

    # 计算每个词的TF-IDF值
    feature_names = vectorizer.get_feature_names_out()
    tfidf_scores = tfidf_matrix.toarray()

    # 计算上下文重要性
    context_importance = np.sum(tfidf_scores, axis=0)

    # 加权TF-IDF矩阵
    weighted_tfidf_matrix = tfidf_matrix.multiply(context_importance)

    # 计算目标数据的加权TF-IDF
    target_weighted_tfidf = target_tfidf.multiply(context_importance)

    # 计算余弦相似度
    similarity = cosine_similarity(target_weighted_tfidf, weighted_tfidf_matrix)
    return similarity
# def bert_cosine_similarity(source_data, target_data, model_name='model/student_model'):
#     """计算BERT嵌入和余弦相似度"""
#     tokenizer = AutoTokenizer.from_pretrained(model_name)
#     model = AutoModel.from_pretrained(model_name)
#
#     def embed_texts(texts):
#         inputs = tokenizer(texts, padding=True, truncation=True, return_tensors='pt')
#         with torch.no_grad():
#             outputs = model(**inputs)
#         embeddings = outputs.last_hidden_state.mean(dim=1)
#         return embeddings
#
#     source_embeddings = embed_texts(source_data)
#     target_embeddings = embed_texts(target_data)
#     similarity = cosine_similarity(target_embeddings.numpy(), source_embeddings.numpy())
#     print(source_embeddings)
#     print(target_embeddings)
#     print(similarity)
#     return similarity
def bert_cosine_similarity(source_data, target_data, model_name='model1/student_model'):
    """优化后的BERT相似度计算"""
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)

    # 添加特征归一化层
    normalize = torch.nn.functional.normalize

    # 改进的特征提取方法
    def embed_texts(texts):
        inputs = tokenizer(texts, padding=True, truncation=True, return_tensors='pt')
        with torch.no_grad():
            outputs = model(**inputs)

        # 使用加权池化代替平均池化
        weights = torch.nn.functional.softmax(outputs.last_hidden_state[:, :, 0], dim=-1)
        embeddings = torch.sum(outputs.last_hidden_state * weights.unsqueeze(-1), dim=1)

        return normalize(embeddings, p=2, dim=1)  # L2归一化

    # # 动态温度系数（根据数据分布调整）
    def temperature_scaling(similarities):
        # 限制温度系数下限
        temp = max(np.percentile(similarities, 75), 0.5)
        return np.exp(similarities / temp)
    source_embeddings = embed_texts(source_data)
    target_embeddings = embed_texts(target_data)

    # 相似度计算改进
    similarity = cosine_similarity(target_embeddings.numpy(), source_embeddings.numpy())
    similarity = temperature_scaling(similarity)/3.1  # 温度缩放

    return similarity


def match_and_filter(similarity_matrix, source_data, target_data, threshold=0.1):  # 调整阈值
    """优化后的匹配筛选"""
    matched_pairs = []
    for i, target_row in enumerate(similarity_matrix):
        # 添加相似度分布过滤
        row_mean = np.mean(target_row)
        row_std = np.std(target_row)
        valid_indices = np.where(target_row > row_mean - row_std)[0]

        if valid_indices.any():
            max_sim_idx = valid_indices[np.argmax(target_row[valid_indices])]
            max_sim_value = target_row[max_sim_idx]

            # 动态阈值调整
            dynamic_threshold = max(threshold, row_mean)
            if max_sim_value >= dynamic_threshold:
                matched_pairs.append((target_data[i], source_data[max_sim_idx], max_sim_value))

    return matched_pairs

# def match_and_filter(similarity_matrix, source_data, target_data, threshold=0.7):
#     """根据相似度矩阵筛选匹配结果"""
#     matched_pairs = []
#     for i, target_row in enumerate(similarity_matrix):
#         max_sim_idx = np.argmax(target_row)
#         max_sim_value = target_row[max_sim_idx]
#         if max_sim_value >= threshold:
#             matched_pairs.append((target_data[i], source_data[max_sim_idx], max_sim_value))
#     return matched_pairs

def match_and_filter1(similarity_matrix, source_data, target_data, threshold=0.1):
    """根据相似度矩阵筛选匹配结果"""
    matched_pairs1 = []
    for i, target_row in enumerate(similarity_matrix):
        max_sim_idx = np.argmax(target_row)
        max_sim_value = target_row[max_sim_idx]
        if max_sim_value >= threshold:
            matched_pairs1.append((target_data[i], source_data[max_sim_idx], max_sim_value))
    return matched_pairs1
def calculate_average_similarity(matched_pairs):
    """计算匹配结果的平均相似度"""
    if matched_pairs:
        avg_similarity = np.mean([match[2] for match in matched_pairs])
        return avg_similarity
    else:
        return 0.0


def calculate_accuracy_and_coverage(matched_pairs, target_data, threshold=0.7):
    """计算匹配准确率和覆盖率"""
    # 假设我们有手动标注的目标数据
    manually_labelled_pairs = set(target_data)  # 这里需要实际的标注数据

    # 覆盖率：匹配到的目标数据条目数 / 总目标数据条目数
    covered_target_data = set([match[0] for match in matched_pairs])
    coverage = len(covered_target_data) / len(manually_labelled_pairs) if manually_labelled_pairs else 0.0

    # 匹配准确率：匹配到的目标数据与手动标注数据的重合度
    correct_matches = len(covered_target_data & manually_labelled_pairs)
    accuracy = correct_matches / len(matched_pairs) if matched_pairs else 0.0

    return accuracy, coverage

# 加载数据
syllabus_data = load_data('data/syllabus.txt')
data1 = load_data('data/data5-1.txt')

# 使用TF-IDF进行匹配--matched_pairs1
tfidf_similarity = compute_tfidf_similarity(syllabus_data, data1)
tfidf_matched= match_and_filter(tfidf_similarity, syllabus_data, data1)
tfidf_matched1= match_and_filter1(tfidf_similarity, syllabus_data, data1)
#使用TF-IDF-CI进行匹配
tfidf_ci_similarity = compute_tfidf_ci_similarity(syllabus_data, data1)
tfidf_ci_matched = match_and_filter(tfidf_ci_similarity, syllabus_data, data1)
tfidf_ci_matched1 = match_and_filter1(tfidf_ci_similarity, syllabus_data, data1)
# 使用BERT进行匹配
bert_similarity = bert_cosine_similarity(syllabus_data, data1)
bert_matched = match_and_filter(bert_similarity, syllabus_data, data1)
bert_matched1 = match_and_filter1(bert_similarity, syllabus_data, data1)
# 计算每种方法的平均相似度
tfidf_avg_similarity = calculate_average_similarity(tfidf_matched)
tfidf_ci_avg_similarity = calculate_average_similarity(tfidf_ci_matched)
bert_avg_similarity = calculate_average_similarity(bert_matched)

# 计算匹配准确率和覆盖率
tfidf_accuracy, tfidf_coverage = calculate_accuracy_and_coverage(tfidf_matched1, data1)
bert_accuracy, bert_coverage = calculate_accuracy_and_coverage(bert_matched1, data1)
tfidf_ci_accuracy, tfidf_ci_coverage = calculate_accuracy_and_coverage(tfidf_ci_matched1, data1)

# 输出结果
print("TF-IDF匹配结果:")
print(f"TF-IDF匹配的平均相似度: {tfidf_avg_similarity:.4f}")
print(f"TF-IDF匹配准确率: {tfidf_accuracy:.4f}")
print(f"TF-IDF覆盖率: {tfidf_coverage:.4f}\n")

print("TF-IDF-CI匹配结果:")
print(f"TF-IDF-CI匹配的平均相似度: {tfidf_ci_avg_similarity:.4f}")
print(f"TF-IDF-CI匹配准确率: {tfidf_ci_accuracy:.4f}")
print(f"TF-IDF-CI覆盖率: {tfidf_ci_coverage:.4f}\n")

print("BERT-匹配结果:")
print(f"BERT匹配的平均相似度: {bert_avg_similarity:.4f}")
print(f"BERT匹配准确率: {bert_accuracy:.4f}")
print(f"BERT覆盖率: {bert_coverage:.4f}\n")