import torch
import torch.nn as nn
import torch.optim as optim

class Pizda_a_ne_Model(nn.Module):
    def __init__(self, input_size, hidden_size, memory_size, memory_dim):
        super(Pizda_a_ne_Model, self).__init__()
        self.hidden_size = hidden_size
        self.memory_size = memory_size
        self.memory_dim = memory_dim

        self.controller = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.memory = nn.Parameter(torch.randn(memory_size, memory_dim))
        self.read_weights = nn.Parameter(torch.randn(hidden_size, memory_size))
        self.write_weights = nn.Parameter(torch.randn(hidden_size, memory_size))
        self.output_layer = nn.Linear(hidden_size + memory_dim, 1)

    def forward(self, x):
        h, _ = self.controller(x)
        h = h[:, -1, :]

        read_weights = torch.softmax(torch.matmul(h, self.read_weights), dim=1)
        read_vector = torch.matmul(read_weights, self.memory)

        write_weights = torch.softmax(torch.matmul(h, self.write_weights), dim=1)
        write_vector = torch.matmul(write_weights.unsqueeze(2), h.unsqueeze(1)).squeeze(1)

        updated_memory = self.memory + write_vector.mean(dim=0)

        combined = torch.cat((h, read_vector), dim=1)
        output = self.output_layer(combined)
        return output, updated_memory

X = torch.tensor([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=torch.float32)
y = torch.tensor([[0], [1], [1], [0]], dtype=torch.float32)

input_size = 2
hidden_size = 4
memory_size = 4
memory_dim = 4

model = Pizda_a_ne_Model(input_size, hidden_size, memory_size, memory_dim)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

num_epochs = 1000
for epoch in range(num_epochs):
    model.train()
    optimizer.zero_grad()
    outputs, updated_memory = model(X.unsqueeze(1))
    loss = criterion(outputs, y)
    loss.backward()
    optimizer.step()
    model.memory.data.copy_(updated_memory)

    if (epoch+1) % 100 == 0:
        print(f'Эпоха [{epoch+1}/{num_epochs}], Потери: {loss.item():.4f}')

model.eval()
with torch.no_grad():
    test_outputs, _ = model(X.unsqueeze(1))
    print('Предсказано:', test_outputs.view(-1).numpy())
    print('Фактически:', y.view(-1).numpy())
