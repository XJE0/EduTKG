import torch
import torch.nn as nn
from transformers import BertModel
from torchcrf import CRF
from sklearn.metrics import classification_report
from data_preprocessing import get_dataloaders
from transformers import BertTokenizer
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

# 定义类别权重，忽略 O 类别的损失计算
class_weights = torch.tensor(
    [1, 4.0, 4.0, 4.0, 4.0, 8.0, 8.0, 4.0, 4.0, 6.0, 6.0],
    dtype=torch.float32
).to('cuda' if torch.cuda.is_available() else 'cpu')


class BiLSTMCRF(nn.Module):
    def __init__(self, bert_model_name, num_labels):
        super(BiLSTMCRF, self).__init__()
        self.bert = BertModel.from_pretrained(bert_model_name)
        self.lstm = nn.LSTM(input_size=self.bert.config.hidden_size, hidden_size=256, num_layers=2, bidirectional=True, batch_first=True)
        self.fc = nn.Linear(256 * 2, num_labels)  # BiLSTM outputs 2 * 256
        self.crf = CRF(num_labels, batch_first=True)

    def forward(self, input_ids, attention_mask, labels=None):
        bert_outputs = self.bert(input_ids, attention_mask=attention_mask)
        sequence_output = bert_outputs.last_hidden_state
        lstm_out, _ = self.lstm(sequence_output)
        logits = self.fc(lstm_out)

        # 应用类别权重
        weighted_logits = logits * class_weights  # 将类别权重应用到 logits

        # 确保 logits 的维度为 (batch_size, seq_len, num_labels)
        logits = weighted_logits.view(weighted_logits.size(0), weighted_logits.size(1), -1)

        if labels is not None:
            # 将填充标签位置设置为不参与损失计算 (-100)
            mask = (labels != -100).byte()
            # 使用 CRF 损失计算
            loss = -self.crf(logits, labels, mask=(labels != -100) & mask.byte())
            return loss
        else:
            # CRF 解码（用于推理）
            predicted_labels = self.crf.decode(logits, mask=(labels != -100) & attention_mask.byte())
            return predicted_labels

def train_bilstm_crf(train_file, eval_file, label_map, batch_size=16, epochs=20, learning_rate=1e-5):
    tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')
    train_loader, eval_loader = get_dataloaders(train_file, eval_file, tokenizer, label_map, batch_size)
    print(f"训练集长度: {len(train_loader.dataset)}")
    print(f"验证集长度: {len(eval_loader.dataset)}")

    model = BiLSTMCRF(bert_model_name="bert-base-chinese", num_labels=len(label_map))
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    model.to('cuda' if torch.cuda.is_available() else 'cpu')

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for batch in train_loader:
            input_ids = batch['input_ids'].to('cuda' if torch.cuda.is_available() else 'cpu')
            attention_mask = batch['attention_mask'].to('cuda' if torch.cuda.is_available() else 'cpu')
            labels = batch['labels'].to('cuda' if torch.cuda.is_available() else 'cpu')
            optimizer.zero_grad()
            loss = model(input_ids, attention_mask, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        print(f"Epoch {epoch + 1}/{epochs} - Loss: {total_loss / len(train_loader)}")
    evaluate_bilstm_crf(model, eval_loader, label_map)

def evaluate_bilstm_crf(model, eval_loader, label_map, save_path='eval/eval_results_bilstm_crf.txt'):
    model.eval()
    true_labels = []
    pred_labels = []

    with torch.no_grad():
        for batch in eval_loader:
            input_ids = batch['input_ids'].to('cuda' if torch.cuda.is_available() else 'cpu')
            attention_mask = batch['attention_mask'].to('cuda' if torch.cuda.is_available() else 'cpu')
            labels = batch['labels'].to('cuda' if torch.cuda.is_available() else 'cpu')

            # 获取模型预测的标签
            predictions = model(input_ids, attention_mask)

            # 将标签和预测结果展平成一维数组以便评估
            for label_seq, pred_seq in zip(labels.cpu().numpy(), predictions):
                # 过滤掉填充的标签 (-100) 和标签0
                true_labels.extend([label for label in label_seq if label != -100])
                pred_labels.extend([pred for label, pred in zip(label_seq, pred_seq) if label != -100])

    # 获取实际出现的标签
    unique_labels = list(set(true_labels))
    precision, recall, f1, _ = precision_recall_fscore_support(true_labels, pred_labels, average='weighted')
    acc = accuracy_score(true_labels, pred_labels)
    print(acc, precision, recall, f1)

    # 保存评估结果到文件
    with open(save_path, 'w', encoding='utf-8') as f:
        f.write(classification_report(true_labels, pred_labels, labels=unique_labels, target_names=[k for k, v in label_map.items() if v in unique_labels]))

if __name__ == "__main__":
    label_map = {
        'O': 0, 'B-tech': 1, 'I-tech': 2, 'B-function': 3, 'I-function': 4,
        'B-chapter': 5, 'I-chapter': 6, 'B-symbol': 7, 'I-symbol': 8,
        'B-kd': 9, 'I-kd': 10
    }
    train_bilstm_crf('data/train.txt', 'data/eval.txt', label_map)
