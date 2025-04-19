# import torch
# from torch.utils.data import Dataset, DataLoader
# from transformers import BertTokenizer, BertModel, AdamW
# from sklearn.model_selection import train_test_split
# import json
# import numpy as np
#
# # 1. 数据预处理
# with open('newdata/data.json', encoding='utf-8') as f:  # 添加编码参数
#     data = json.load(f)
# # 定义关系标签映射
# relations = [
#     '包含', '关联', '使用', '区别于',
#     '实现方式',
# ]
# rel2id = {rel: i for i, rel in enumerate(relations)}
#
# # 构建训练样本
# samples = []
# for item in data:
#     text = f"{item['头实体'][3]} {item['关系'][0]} {item['尾实体'][3]}"
#     head_ent = item['头实体'][3]
#     tail_ent = item['尾实体'][3]
#     relation = item['关系'][0]
#
#     # 添加实体位置标记
#     marked_text = text.replace(head_ent, f"[E1]{head_ent}[/E1]")
#     marked_text = marked_text.replace(tail_ent, f"[E2]{tail_ent}[/E2]")
#
#     samples.append({
#         'text': marked_text,
#         'head': head_ent,
#         'tail': tail_ent,
#         'relation': rel2id[relation]
#     })
#
#
# # 2. 自定义Dataset
# class RelationDataset(Dataset):
#     def __init__(self, data, tokenizer, max_len=128):
#         self.data = data
#         self.tokenizer = tokenizer
#         self.max_len = max_len
#
#     def __len__(self):
#         return len(self.data)
#
#     def __getitem__(self, idx):
#         sample = self.data[idx]
#         text = sample['text']
#
#         # 定位实体标记
#         e1_start = text.find('[E1]') + 4
#         e1_end = text.find('[/E1]')
#         e2_start = text.find('[E2]') + 4
#         e2_end = text.find('[/E2]')
#
#         # 编码文本
#         encoding = self.tokenizer.encode_plus(
#             text,
#             add_special_tokens=True,
#             max_length=self.max_len,
#             padding='max_length',
#             truncation=True,
#             return_attention_mask=True,
#             return_tensors='pt'
#         )
#
#         # 创建实体位置特征
#         e1_mask = torch.zeros(self.max_len)
#         e2_mask = torch.zeros(self.max_len)
#
#         for i, (input_id, token) in enumerate(zip(encoding['input_ids'][0],
#                                                   self.tokenizer.convert_ids_to_tokens(encoding['input_ids'][0]))):
#             if e1_start <= i < e1_end:
#                 e1_mask[i] = 1
#             if e2_start <= i < e2_end:
#                 e2_mask[i] = 1
#
#         return {
#             'input_ids': encoding['input_ids'].flatten(),
#             'attention_mask': encoding['attention_mask'].flatten(),
#             'e1_mask': e1_mask,
#             'e2_mask': e2_mask,
#             'label': torch.tensor(sample['relation'], dtype=torch.long)
#         }
#
#
# # 3. 模型架构
# class BERTRelationClassifier(torch.nn.Module):
#     def __init__(self, n_classes):
#         super().__init__()
#         self.bert = BertModel.from_pretrained('bert-base-chinese')
#         self.dropout = torch.nn.Dropout(0.3)
#         self.classifier = torch.nn.Linear(self.bert.config.hidden_size * 3, n_classes)
#
#     def forward(self, input_ids, attention_mask, e1_mask, e2_mask):
#         outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
#         last_hidden_state = outputs.last_hidden_state
#
#         # 提取实体特征
#         e1_features = (last_hidden_state * e1_mask.unsqueeze(-1)).sum(dim=1)
#         e2_features = (last_hidden_state * e2_mask.unsqueeze(-1)).sum(dim=1)
#
#         # 组合特征
#         combined = torch.cat([
#             last_hidden_state[:, 0],  # [CLS] token
#             e1_features,
#             e2_features
#         ], dim=1)
#
#         output = self.dropout(combined)
#         return self.classifier(output)
#
#
# # 4. 训练配置
# tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')
# train_data, val_data = train_test_split(samples, test_size=0.2)
#
# BATCH_SIZE = 16
# EPOCHS = 10
# LEARNING_RATE = 2e-5
#
# train_dataset = RelationDataset(train_data, tokenizer)
# val_dataset = RelationDataset(val_data, tokenizer)
#
# train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
# val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)
#
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# model = BERTRelationClassifier(len(relations)).to(device)
# optimizer = AdamW(model.parameters(), lr=LEARNING_RATE)
# criterion = torch.nn.CrossEntropyLoss()
# model.bert.resize_token_embeddings(len(tokenizer))  # 调整模型embeddings大小
# # 5. 训练循环
# for epoch in range(EPOCHS):
#     model.train()
#     total_loss = 0
#
#     for batch in train_loader:
#         optimizer.zero_grad()
#
#         input_ids = batch['input_ids'].to(device)
#         attention_mask = batch['attention_mask'].to(device)
#         e1_mask = batch['e1_mask'].to(device)
#         e2_mask = batch['e2_mask'].to(device)
#         labels = batch['label'].to(device)
#
#         outputs = model(input_ids, attention_mask, e1_mask, e2_mask)
#         loss = criterion(outputs, labels)
#         total_loss += loss.item()
#
#         loss.backward()
#         optimizer.step()
#
#     avg_train_loss = total_loss / len(train_loader)
#
#     # 验证
#     model.eval()
#     val_loss = 0
#     correct = 0
#     total = 0
#
#     with torch.no_grad():
#         for batch in val_loader:
#             input_ids = batch['input_ids'].to(device)
#             attention_mask = batch['attention_mask'].to(device)
#             e1_mask = batch['e1_mask'].to(device)
#             e2_mask = batch['e2_mask'].to(device)
#             labels = batch['label'].to(device)
#
#             outputs = model(input_ids, attention_mask, e1_mask, e2_mask)
#             loss = criterion(outputs, labels)
#             val_loss += loss.item()
#
#             _, predicted = torch.max(outputs, 1)
#             correct += (predicted == labels).sum().item()
#             total += labels.size(0)
#
#     avg_val_loss = val_loss / len(val_loader)
#     accuracy = correct / total
#
#     print(f"Epoch {epoch + 1}/{EPOCHS}")
#     print(f"Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | Val Acc: {accuracy:.4f}")
#
#
# # 6. 推理函数
# # 修改预测函数，修复实体定位问题
# def predict_relation(text, head_entity, tail_entity):
#     # 添加特殊标记到tokenizer（关键修复）
#     entity_tags = ['[E1]', '[/E1]', '[E2]', '[/E2]']
#     tokenizer.add_special_tokens({'additional_special_tokens': entity_tags})
#
#     # 安全替换实体（防止多次替换）
#     marked_text = text.replace(head_entity, f" [E1]{head_entity}[/E1] ", 1)
#     marked_text = marked_text.replace(tail_entity, f" [E2]{tail_entity}[/E2] ", 1)
#
#     # 编码处理
#     encoding = tokenizer.encode_plus(
#         marked_text,
#         add_special_tokens=True,
#         max_length=128,
#         padding='max_length',
#         truncation=True,
#         return_tensors='pt'
#     )
#
#     # 初始化默认位置
#     e1_start, e1_end = 0, 0
#     e2_start, e2_end = 0, 0
#     tokens = tokenizer.convert_ids_to_tokens(encoding['input_ids'][0])
#
#     # 安全定位实体标记
#     try:
#         e1_start = tokens.index('[E1]') + 1  # +1跳过起始标记
#         e1_end = tokens.index('[/E1]')
#     except ValueError:
#         print(f"警告：未找到头实体标记 [E1]")
#
#     try:
#         e2_start = tokens.index('[E2]') + 1
#         e2_end = tokens.index('[/E2]')
#     except ValueError:
#         print(f"警告：未找到尾实体标记 [E2]")
#
#     # 创建mask张量
#     e1_mask = torch.zeros(128)
#     e2_mask = torch.zeros(128)
#     e1_mask[e1_start:e1_end] = 1
#     e2_mask[e2_start:e2_end] = 1
#
#     # 模型预测
#     model.eval()
#     with torch.no_grad():
#         input_ids = encoding['input_ids'].to(device)
#         attention_mask = encoding['attention_mask'].to(device)
#         e1_mask = e1_mask.unsqueeze(0).to(device)
#         e2_mask = e2_mask.unsqueeze(0).to(device)
#
#         output = model(input_ids, attention_mask, e1_mask, e2_mask)
#         _, prediction = torch.max(output, dim=1)
#
#     return relations[prediction.item()]
#
#
#
#
#
# # 使用示例
# text = "在C语言中，指针可以用来实现链表数据结构"
# head = "指针"
# tail = "链表"
# print(predict_relation(text, head, tail))  # 输出：实现基础
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertModel, AdamW
from sklearn.model_selection import train_test_split
import json
import re

# 1. 增强数据预处理
with open('newdata/data.json', encoding='utf-8') as f:
    data = json.load(f)

# 定义关系标签映射
relations = ['包含', '组合', '使用',]
rel2id = {rel: i for i, rel in enumerate(relations)}

# 构建训练样本（带数据验证）
samples = []
for item in data:
    try:
        # 验证数据完整性
        assert "头实体" in item and len(item["头实体"]) >= 4
        assert "尾实体" in item and len(item["尾实体"]) >= 4
        assert "关系" in item and len(item["关系"]) >= 1

        head_ent = item["头实体"][3]
        tail_ent = item["尾实体"][3]
        relation = item["关系"][0]

        # 使用正则表达式安全替换实体
        base_text = f"{head_ent} {relation} {tail_ent}"
        marked_text = re.sub(re.escape(head_ent), f'[E1]{head_ent}[/E1]', base_text, count=1)
        marked_text = re.sub(re.escape(tail_ent), f'[E2]{tail_ent}[/E2]', marked_text, count=1)

        samples.append({
            'text': marked_text,
            'head': head_ent,
            'tail': tail_ent,
            'relation': rel2id[relation]
        })
    except (KeyError, IndexError, AssertionError) as e:
        print(f"无效数据项：{item}，错误：{str(e)}")
        continue


# 2. 改进Dataset类
class RelationDataset(Dataset):
    def __init__(self, data, tokenizer, max_len=128):
        self.data = data
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.tokenizer.add_special_tokens({'additional_special_tokens': ['[E1]', '[/E1]', '[E2]', '[/E2]']})

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        text = sample['text']

        # 基于分词后的token定位实体
        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )

        tokens = self.tokenizer.convert_ids_to_tokens(encoding['input_ids'][0])

        # 初始化mask
        e1_mask = torch.zeros(self.max_len)
        e2_mask = torch.zeros(self.max_len)

        # 定位实体标记
        try:
            e1_start = tokens.index('[E1]') + 1
            e1_end = tokens.index('[/E1]')
            e1_mask[e1_start:e1_end] = 1
        except ValueError:
            pass

        try:
            e2_start = tokens.index('[E2]') + 1
            e2_end = tokens.index('[/E2]')
            e2_mask[e2_start:e2_end] = 1
        except ValueError:
            pass

        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'e1_mask': e1_mask,
            'e2_mask': e2_mask,
            'label': torch.tensor(sample['relation'], dtype=torch.long)
        }


# 3. 模型架构（保持不变）
class BERTRelationClassifier(torch.nn.Module):
    def __init__(self, n_classes):
        super().__init__()
        self.bert = BertModel.from_pretrained('bert-base-chinese')
        self.dropout = torch.nn.Dropout(0.3)
        self.classifier = torch.nn.Linear(self.bert.config.hidden_size * 3, n_classes)

    def forward(self, input_ids, attention_mask, e1_mask, e2_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        last_hidden_state = outputs.last_hidden_state

        e1_features = (last_hidden_state * e1_mask.unsqueeze(-1)).sum(dim=1)
        e2_features = (last_hidden_state * e2_mask.unsqueeze(-1)).sum(dim=1)

        combined = torch.cat([
            last_hidden_state[:, 0],
            e1_features,
            e2_features
        ], dim=1)

        output = self.dropout(combined)
        return self.classifier(output)


# 4. 训练配置
tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')
train_data, val_data = train_test_split(samples, test_size=0.2)

BATCH_SIZE = 16
EPOCHS = 10
LEARNING_RATE = 2e-5

train_dataset = RelationDataset(train_data, tokenizer)
val_dataset = RelationDataset(val_data, tokenizer)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = BERTRelationClassifier(len(relations)).to(device)
model.bert.resize_token_embeddings(len(tokenizer))  # 扩展词表

optimizer = AdamW(model.parameters(), lr=LEARNING_RATE)
criterion = torch.nn.CrossEntropyLoss()


# 5. 训练循环（保持不变）
for epoch in range(EPOCHS):
    model.train()
    total_loss = 0

    for batch in train_loader:
        optimizer.zero_grad()

        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        e1_mask = batch['e1_mask'].to(device)
        e2_mask = batch['e2_mask'].to(device)
        labels = batch['label'].to(device)

        outputs = model(input_ids, attention_mask, e1_mask, e2_mask)
        loss = criterion(outputs, labels)
        total_loss += loss.item()

        loss.backward()
        optimizer.step()

    avg_train_loss = total_loss / len(train_loader)

    # 验证
    model.eval()
    val_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():
        for batch in val_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            e1_mask = batch['e1_mask'].to(device)
            e2_mask = batch['e2_mask'].to(device)
            labels = batch['label'].to(device)

            outputs = model(input_ids, attention_mask, e1_mask, e2_mask)
            loss = criterion(outputs, labels)
            val_loss += loss.item()

            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

    avg_val_loss = val_loss / len(val_loader)
    accuracy = correct / total

    print(f"Epoch {epoch + 1}/{EPOCHS}")
    print(f"Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | Val Acc: {accuracy:.4f}")


# 6. 增强的推理函数
def predict_relation(text, head_entity, tail_entity):
    # 安全替换实体
    marked_text = re.sub(re.escape(head_entity), f'[E1]{head_entity}[/E1]', text, count=1)
    marked_text = re.sub(re.escape(tail_entity), f'[E2]{tail_entity}[/E2]', marked_text, count=1)

    # 编码处理
    encoding = tokenizer.encode_plus(
        marked_text,
        add_special_tokens=True,
        max_length=128,
        padding='max_length',
        truncation=True,
        return_tensors='pt'
    )

    # 基于分词定位实体
    tokens = tokenizer.convert_ids_to_tokens(encoding['input_ids'][0])

    # 初始化mask
    e1_mask = torch.zeros(128)
    e2_mask = torch.zeros(128)

    try:
        e1_start = tokens.index('[E1]') + 1
        e1_end = tokens.index('[/E1]')
        e1_mask[e1_start:e1_end] = 1
    except ValueError:
        print("警告：未找到头实体标记")

    try:
        e2_start = tokens.index('[E2]') + 1
        e2_end = tokens.index('[/E2]')
        e2_mask[e2_start:e2_end] = 1
    except ValueError:
        print("警告：未找到尾实体标记")

    # 模型预测
    model.eval()
    with torch.no_grad():
        inputs = {
            'input_ids': encoding['input_ids'].to(device),
            'attention_mask': encoding['attention_mask'].to(device),
            'e1_mask': e1_mask.unsqueeze(0).to(device),
            'e2_mask': e2_mask.unsqueeze(0).to(device)
        }
        output = model(**inputs)
        _, prediction = torch.max(output, dim=1)

    return relations[prediction.item()]


# 使用示例
text = "在C语言中，指针可以用来实现链表数据结构"
head = "指针"
tail = "链表"
print(predict_relation(text, head, tail))  # 输出：实现基础