import nltk
import torch
from data_preprocessing import get_dataloaders
from torch.utils.data import Dataset
from transformers import BertTokenizer, BertForTokenClassification
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import torch.nn as nn
from sklearn.metrics import classification_report

nltk.download('punkt')

label_map = {
    'O': 0,
    'B-tech': 1,
    'I-tech': 2,
    'B-function': 3,
    'I-function': 4,
    'B-chapter': 5,
    'I-chapter': 6,
    'B-symbol': 7,
    'I-symbol': 8,
    'B-kd': 9,
    'I-kd': 10
}

# 定义类别权重，忽略 O 类别的损失计算
class_weights = torch.tensor(
    [1, 4.0, 4.0, 4.0, 4.0, 8.0, 8.0, 4.0, 4.0, 6.0, 6.0],
    dtype=torch.float32
).to('cuda' if torch.cuda.is_available() else 'cpu')

class NERDataset(Dataset):
    def __init__(self, sentences, labels, tokenizer, max_len=128):
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.sentences = sentences
        self.labels = labels

    def __len__(self):
        return len(self.sentences)

    def __getitem__(self, idx):
        sentence = self.sentences[idx]
        labels = self.labels[idx]

        encoding = self.tokenizer(
            sentence, is_split_into_words=True, truncation=True,
            padding='max_length', max_length=self.max_len, return_tensors='pt'
        )

        label_ids = [-100] * self.max_len
        for i, label in enumerate(labels):
            if i < self.max_len:
                label_ids[i] = label_map.get(label, -100)

        return {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'labels': torch.tensor(label_ids, dtype=torch.long)
        }

# 动态伪标签生成
def generate_dynamic_pseudo_labels(sentences, model, tokenizer, device, confidence_threshold=0.89):
    inputs = tokenizer(sentences, padding=True, truncation=True, return_tensors='pt').to(device)

    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        max_probs, max_labels = logits.softmax(dim=-1).max(dim=-1)

    pseudo_sentences, pseudo_labels = [], []
    for i, sentence in enumerate(sentences):
        sentence_labels = []
        include_sentence = False
        for j, label_id in enumerate(max_labels[i]):
            if max_probs[i][j] > confidence_threshold:
                sentence_labels.append(list(label_map.keys())[label_id.item()])
                include_sentence = True
            else:
                sentence_labels.append('O')

        if include_sentence:
            pseudo_sentences.append(sentence)
            pseudo_labels.append(sentence_labels)

    return pseudo_sentences, pseudo_labels

# 数据读取和处理
def read_data(file_path):
    sentences = []
    labels = []
    sentence = []
    label = []

    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            if line.strip() == "":
                if sentence:
                    sentences.append(sentence)
                    labels.append(label)
                    sentence = []
                    label = []
            else:
                word, tag = line.strip().split()
                sentence.append(word)
                label.append(tag)

        if sentence:
            sentences.append(sentence)
            labels.append(label)

    return sentences, labels

def save_combined_dataset_to_txt(sentences, labels, output_file):
    with open(output_file, 'w', encoding='utf-8') as f:
        for sentence, label in zip(sentences, labels):
            for word, tag in zip(sentence, label):
                f.write(f"{word} {tag}\n")
            f.write("\n")

def main():
    tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')
    teacher_model = BertForTokenClassification.from_pretrained('bert-base-chinese', num_labels=len(label_map))
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    teacher_model.to(device)

    # 读取文本数据
    def read_txt(file_path):
        with open(file_path, 'r', encoding='utf-8') as f:
            return f.read()

    def split_text_into_sentences(text):
        return nltk.sent_tokenize(text)

    text = read_txt('data/data3.txt')
    sentences = split_text_into_sentences(text)

    original_sentences, original_labels = read_data('data/train.txt')

    # 动态训练：多轮迭代，每轮更新伪标签
    max_cycles = 1
    for cycle in range(max_cycles):
        print(f"\n--- 第 {cycle + 1} 轮动态训练 ---")

        # 生成伪标签
        pseudo_sentences, pseudo_labels = generate_dynamic_pseudo_labels(sentences, teacher_model, tokenizer, device, confidence_threshold=0.5)

        # 合并原始数据和伪标签数据
        combined_sentences = original_sentences + pseudo_sentences
        combined_labels = original_labels + pseudo_labels

        # 保存每轮的合并数据集
        combined_dataset_path = f'data/combined_dataset_cycle_{cycle + 1}.txt'
        save_combined_dataset_to_txt(combined_sentences, combined_labels, combined_dataset_path)

        # 初始化学生模型
        student_model = BertForTokenClassification.from_pretrained('bert-base-chinese', num_labels=len(label_map))
        student_model.to(device)

        optimizer = torch.optim.AdamW(student_model.parameters(), lr=1e-5)
        epochs = 40
        batch_size = 16

        combined_dataset = NERDataset(combined_sentences, combined_labels, tokenizer)
        train_loader = torch.utils.data.DataLoader(combined_dataset, batch_size=batch_size, shuffle=True)

        # 训练学生模型
        for epoch in range(epochs):
            student_model.train()
            total_loss = 0

            for batch in train_loader:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['labels'].to(device)

                optimizer.zero_grad()

                outputs = student_model(input_ids, attention_mask, labels=labels)
                loss = outputs.loss

                weights = class_weights[labels]
                weights[labels == -100] = 0

                weighted_loss = (loss * weights).mean()

                weighted_loss.backward()
                optimizer.step()
                total_loss += weighted_loss.item()

            print(f"Cycle {cycle + 1}, Epoch {epoch + 1}/{epochs} - Loss: {total_loss / len(train_loader)}")

        # 保存每轮的学生模型
        # student_model_path = f'model/student_model_cycle_{cycle + 1}'
        # student_model.save_pretrained(student_model_path)
        # tokenizer.save_pretrained(student_model_path)
        # 更新教师模型
        teacher_model = student_model
    print("动态训练完成，所有轮次模型已保存！")
    train_loader, eval_loader = get_dataloaders('data/train.txt', 'data/eval.txt', tokenizer, label_map, batch_size=16)
    evaluate_bert_ner(student_model, eval_loader, label_map)


def evaluate_bert_ner(model, eval_loader, label_map, save_path='eval/eval_results_bert_self_training.txt'):
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
                true_labels.extend([label for label in label_seq])  # 过滤掉标签为0
                pred_labels.extend([pred for label, pred in zip(label_seq, pred_seq)])  # 过滤掉标签为0的预测

    # 获取实际出现的标签（排除标签0）
    unique_labels = list(set(true_labels))
    precision, recall, f1, _ = precision_recall_fscore_support(true_labels, pred_labels, average='weighted')
    acc = accuracy_score(true_labels, pred_labels)
    print(acc, precision, recall, f1)
    # 输出分类报告
    print("分类报告:")
    print(classification_report(true_labels, pred_labels, labels=unique_labels,
                                target_names=[k for k, v in label_map.items() if v in unique_labels]))

    # 保存评估结果到文件
    with open(save_path, 'w', encoding='utf-8') as f:
        f.write(classification_report(true_labels, pred_labels, labels=unique_labels,
                                      target_names=[k for k, v in label_map.items() if v in unique_labels]))

if __name__ == "__main__":
    main()
