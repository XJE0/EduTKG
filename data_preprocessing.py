import torch
from torch.utils.data import Dataset
from transformers import BertTokenizer


class NERDataset(Dataset):
    def __init__(self, file_path, tokenizer, label_map, max_length=128):
        self.sentences = []
        self.labels = []
        self.tokenizer = tokenizer
        self.label_map = label_map
        self.max_length = max_length
        self.load_data(file_path)

    def load_data(self, file_path):
        sentence = []
        label = []
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    word, tag = line.strip().split()
                    sentence.append(word)
                    label.append(self.label_map[tag])
                else:
                    if sentence:
                        self.sentences.append(sentence)
                        self.labels.append(label)
                    sentence = []
                    label = []
            # Add last sentence if not added
            if sentence:
                self.sentences.append(sentence)
                self.labels.append(label)

        # Ensure sentences are split into chunks of max_length (128)
        self.split_sentences()

    def split_sentences(self):
        """将每个句子拆分为多个长度为 max_length 的句子，且不足 max_length 的句子不做处理"""
        split_sentences = []
        split_labels = []

        for sentence, label in zip(self.sentences, self.labels):
            # 将句子拆分成多个小块，每块最多 max_length 个单词
            for i in range(0, len(sentence), self.max_length):
                sub_sentence = sentence[i:i + self.max_length]
                sub_label = label[i:i + self.max_length]

                # 仅当子句的长度等于 max_length 时才添加该子句
                if len(sub_sentence) == self.max_length:
                    split_sentences.append(sub_sentence)
                    split_labels.append(sub_label)

        # 更新 sentences 和 labels
        self.sentences = split_sentences
        self.labels = split_labels

    def __len__(self):
        return len(self.sentences)

    def __getitem__(self, idx):
        sentence = self.sentences[idx]
        label = self.labels[idx]

        # Tokenize the sentence
        inputs = self.tokenizer(sentence, is_split_into_words=True, padding='max_length', truncation=True,
                                max_length=self.max_length, return_tensors="pt")
        attention_mask = inputs["attention_mask"].squeeze()
        input_ids = inputs["input_ids"].squeeze()

        # Return the tokenized inputs and attention mask
        # Ensure labels are of the correct type (long)
        labels = torch.tensor(label, dtype=torch.long)

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels
        }


def get_dataloaders(train_file, eval_file, tokenizer, label_map, batch_size=32):
    train_dataset = NERDataset(train_file, tokenizer, label_map)
    eval_dataset = NERDataset(eval_file, tokenizer, label_map)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    eval_loader = torch.utils.data.DataLoader(eval_dataset, batch_size=batch_size)
    return train_loader, eval_loader


from torch.utils.data import DataLoader

# 自定义数据集类
class CustomDataset(Dataset):
    def __init__(self, texts, labels=None, tokenizer=None, max_length=128):
        self.texts = texts  # 输入文本列表
        self.labels = labels  # 标注数据标签
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        # 使用 BERT Tokenizer 对文本进行编码
        encoding = self.tokenizer(text, truncation=True, padding='max_length', max_length=self.max_length,
                                  return_tensors='pt')

        input_ids = encoding['input_ids'].squeeze(0)  # [max_length]
        attention_mask = encoding['attention_mask'].squeeze(0)  # [max_length]

        if self.labels is not None:
            labels = torch.tensor(self.labels[idx])  # 标注标签
            return {'input_ids': input_ids, 'attention_mask': attention_mask, 'labels': labels}
        else:
            return {'input_ids': input_ids, 'attention_mask': attention_mask}

# 获取无标注数据集的 DataLoader
def get_unlabeled_dataloader(unlabeled_file, tokenizer, batch_size=16, max_length=128):
    """
    从无标注文件中加载数据并返回DataLoader
    """
    # 读取无标注文本文件
    with open(unlabeled_file, 'r', encoding='utf-8') as f:
        unlabeled_texts = f.readlines()

    # 清理文本，去除空白行和多余的换行符
    unlabeled_texts = [text.strip() for text in unlabeled_texts if text.strip()]

    # 创建自定义数据集
    unlabeled_dataset = CustomDataset(unlabeled_texts, tokenizer=tokenizer, max_length=max_length)

    # 创建DataLoader
    unlabeled_loader = DataLoader(unlabeled_dataset, batch_size=batch_size, shuffle=False)

    return unlabeled_loader
