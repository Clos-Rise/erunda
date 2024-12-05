import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import json
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split
from collections import Counter
import numpy as np

class TextDataset(Dataset):
    def __init__(self, data, vocab, max_len):
        self.data = data
        self.vocab = vocab
        self.max_len = max_len

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        text = item['text']
        author = item['author']
        label = 0 if author == 'Александр Пушкин' else 1
        tokens = text.split()
        indices = [self.vocab[token] for token in tokens if token in self.vocab]
        if len(indices) < self.max_len:
            indices += [0] * (self.max_len - len(indices))
        else:
            indices = indices[:self.max_len]
        return torch.tensor(indices, dtype=torch.long), torch.tensor(label, dtype=torch.long)

class Mandarin1(nn.Module):
    def __init__(self, mandarin2, mandarin3, mandarin4, mandarin5):
        super(Mandarin1, self).__init__()
        self.mandarin6 = nn.Embedding(mandarin2, mandarin3)
        self.mandarin7 = nn.LSTM(mandarin3, mandarin4, batch_first=True)
        self.mandarin8 = nn.Linear(mandarin4, mandarin5)

    def forward(self, mandarin9):
        mandarin10 = self.mandarin6(mandarin9)
        mandarin10, _ = self.mandarin7(mandarin10)
        mandarin10 = mandarin10[:, -1, :]
        mandarin10 = self.mandarin8(mandarin10)
        return mandarin10

with open('dataset.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

texts = [item['text'] for item in data]
all_tokens = [token for text in texts for token in text.split()]
vocab = Counter(all_tokens)
vocab = {token: i+1 for i, (token, count) in enumerate(vocab.most_common())}
vocab['<PAD>'] = 0
max_len = max(len(text.split()) for text in texts)
train_data, test_data = train_test_split(data, test_size=0.2, random_state=42)

train_dataset = TextDataset(train_data, vocab, max_len)
test_dataset = TextDataset(test_data, vocab, max_len)
train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=4, shuffle=False)

mandarin11 = len(vocab)
mandarin12 = 128
mandarin13 = 64
mandarin14 = 2
mandarin15 = Mandarin1(mandarin11, mandarin12, mandarin13, mandarin14)

mandarin16 = nn.CrossEntropyLoss()
mandarin17 = optim.Adam(mandarin15.parameters(), lr=0.001)

mandarin18 = 10
for mandarin19 in range(mandarin18):
    mandarin15.train()
    total_acc, total_count = 0, 0
    for mandarin20, (mandarin21, mandarin22) in enumerate(train_loader):
        mandarin17.zero_grad()
        mandarin23 = mandarin15(mandarin21)
        mandarin24 = mandarin16(mandarin23, mandarin22)
        mandarin24.backward()
        mandarin17.step()
        total_acc += (mandarin23.argmax(1) == mandarin22).sum().item()
        total_count += mandarin22.size(0)
    print(f'Эпоха [{mandarin19+1}/{mandarin18}], Потери: {mandarin24.item():.4f}, Точность: {total_acc/total_count:.4f}')

mandarin15.eval()
with torch.no_grad():
    total_acc, total_count = 0, 0
    for mandarin21, mandarin22 in test_loader:
        mandarin23 = mandarin15(mandarin21)
        total_acc += (mandarin23.argmax(1) == mandarin22).sum().item()
        total_count += mandarin22.size(0)
    print(f'Точность текста: {total_acc/total_count:.4f}')
