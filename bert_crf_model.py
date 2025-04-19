import torch
import torch.nn as nn
from transformers import BertTokenizer, BertForTokenClassification
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from data_preprocessing import get_dataloaders  # 假设你已有数据处理函数
import os
from torch.utils.data import Dataset
from transformers import BertTokenizer, BertForTokenClassification, Trainer, TrainingArguments


class BertNERModel(nn.Module):
    def __init__(self, bert_model_name, num_labels):
        super(BertNERModel, self).__init__()
        # 加载 BERT 预训练模型
        self.bert = BertForTokenClassification.from_pretrained(bert_model_name, num_labels=num_labels)

    def forward(self, input_ids, attention_mask, labels=None):
        # BERT 模型的输出
        outputs = self.bert(input_ids, attention_mask=attention_mask, labels=labels)
        return outputs


def train_bert_ner(train_file, eval_file, label_map, batch_size=16, epochs=20, learning_rate=1e-5):
    tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')
    train_loader, eval_loader = get_dataloaders(train_file, eval_file, tokenizer, label_map, batch_size)
    print(f"训练集长度: {len(train_loader.dataset)}")
    print(f"验证集长度: {len(eval_loader.dataset)}")

    model = BertNERModel(bert_model_name="bert-base-chinese", num_labels=len(label_map))

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    model.to('cuda' if torch.cuda.is_available() else 'cpu')

    # 定义类别权重，忽略 O 类别的损失计算
    class_weights = torch.tensor(
        [1, 4.0, 4.0, 4.0, 4.0, 8.0, 8.0, 4.0, 4.0, 6.0, 6.0],
        dtype=torch.float32
    ).to('cuda' if torch.cuda.is_available() else 'cpu')

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for batch in train_loader:
            input_ids = batch['input_ids'].to('cuda' if torch.cuda.is_available() else 'cpu')
            attention_mask = batch['attention_mask'].to('cuda' if torch.cuda.is_available() else 'cpu')
            labels = batch['labels'].to('cuda' if torch.cuda.is_available() else 'cpu')

            optimizer.zero_grad()

            # 计算损失
            outputs = model(input_ids, attention_mask, labels=labels)
            loss = outputs.loss  # 这是一个形状为[batch_size, sequence_length]的损失张量

            # 获取每个token的类别权重
            weights = class_weights[labels]  # 获取每个token的类别权重

            # 对标签为-100的部分忽略损失计算
            weights[labels == -100] = 0

            # 计算加权损失
            weighted_loss = (loss * weights).mean()  # 按照每个token的权重来加权损失

            weighted_loss.backward()
            optimizer.step()
            total_loss += weighted_loss.item()

        print(f"Epoch {epoch + 1}/{epochs} - Loss: {total_loss / len(train_loader)}")

    # 保存模型和Tokenizer
    save_model(model, tokenizer)
    # 评估
    evaluate_bert_ner(model, eval_loader, label_map)


def save_model(model, tokenizer, save_path="model/bert_model"):
    os.makedirs(save_path, exist_ok=True)
    # 保存模型的state_dict（权重）
    torch.save(model.state_dict(), f"{save_path}/pytorch_model.bin")
    # 保存Tokenizer
    tokenizer.save_pretrained(save_path)
    print(f"模型和Tokenizer已保存至 {save_path}")


def evaluate_bert_ner(model, eval_loader, label_map, save_path='eval/eval_results_bert.txt'):
    model.eval()
    true_labels = []
    pred_labels = []

    with torch.no_grad():
        for batch in eval_loader:
            input_ids = batch['input_ids'].to('cuda' if torch.cuda.is_available() else 'cpu')
            attention_mask = batch['attention_mask'].to('cuda' if torch.cuda.is_available() else 'cpu')
            labels = batch['labels'].to('cuda' if torch.cuda.is_available() else 'cpu')

            # 获取模型的预测标签
            outputs = model(input_ids, attention_mask)
            logits = outputs.logits
            predictions = torch.argmax(logits, dim=-1)

            # 将标签和预测结果展平成一维数组，过滤掉标签为0的部分
            for label_seq, pred_seq in zip(labels.cpu().numpy(), predictions.cpu().numpy()):
                true_labels.extend([label for label in label_seq if label != -100])  # 过滤掉标签为-100
                pred_labels.extend([pred for label, pred in zip(label_seq, pred_seq) if label != -100])  # 过滤掉标签为-100的预测

    # 获取实际出现的标签（排除标签0）
    unique_labels = list(set(true_labels))
    precision, recall, f1, _ = precision_recall_fscore_support(true_labels, pred_labels, average='weighted')
    acc = accuracy_score(true_labels, pred_labels)
    print(acc, precision, recall, f1)

    # 输出分类报告
    # print("分类报告:")
    # print(classification_report(true_labels, pred_labels, labels=unique_labels,
    #                             target_names=[k for k, v in label_map.items() if v in unique_labels]))

    # 保存评估结果到文件
    with open(save_path, 'w', encoding='utf-8') as f:
        f.write(classification_report(true_labels, pred_labels, labels=unique_labels,
                                      target_names=[k for k, v in label_map.items() if v in unique_labels]))


if __name__ == "__main__":
    label_map = {
        'O': 0, 'B-tech': 1, 'I-tech': 2, 'B-function': 3, 'I-function': 4,
        'B-chapter': 5, 'I-chapter': 6, 'B-symbol': 7, 'I-symbol': 8,
        'B-kd': 9, 'I-kd': 10
    }
    relatin_lap={'biaohan':0,'brother':1,'zuhe':2,'wuguan':3}
    train_bert_ner('data/train.txt', 'data/eval.txt', label_map)
