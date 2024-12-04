import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import math
from torch.optim.optimizer import Optimizer

class ADAM_i_eva(Optimizer):
    def __init__(self, params, arbuz1=1e-3, arbuz2=0.9, arbuz3=0.999, arbuz4=1e-8, arbuz5=0.9):
        if not 0.0 <= arbuz1:
            raise ValueError("Инвалид learning rate: {}".format(arbuz1))
        if not 0.0 <= arbuz4:
            raise ValueError("Инвалид epsilon value: {}".format(arbuz4))
        if not 0.0 <= arbuz2 < 1.0:
            raise ValueError("Инвалид beta1 parameter: {}".format(arbuz2))
        if not 0.0 <= arbuz3 < 1.0:
            raise ValueError("Инвалид beta2 parameter: {}".format(arbuz3))
        if not 0.0 <= arbuz5 < 1.0:
            raise ValueError("Инвалид momentum parameter: {}".format(arbuz5))

        defaults = dict(arbuz1=arbuz1, arbuz2=arbuz2, arbuz3=arbuz3, arbuz4=arbuz4, arbuz5=arbuz5)
        super(ADAM_i_eva, self).__init__(params, defaults)

    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data
                if grad.is_sparse:
                    raise RuntimeError('Arbuz_Optimizer does not support sparжа gradients')

                state = self.state[p]

                if len(state) == 0:
                    state['arbuz6'] = 0
                    state['arbuz7'] = torch.zeros_like(p.data)
                    state['arbuz8'] = torch.zeros_like(p.data)
                    state['arbuz9'] = torch.zeros_like(p.data)

                arbuz7, arbuz8 = state['arbuz7'], state['arbuz8']
                arbuz9 = state['arbuz9']
                arbuz2, arbuz3 = group['arbuz2'], group['arbuz3']

                state['arbuz6'] += 1

                arbuz7.mul_(arbuz2).add_(grad, alpha=1 - arbuz2)
                arbuz8.mul_(arbuz3).addcmul_(grad, grad, value=1 - arbuz3)
                denom = arbuz8.sqrt().add_(group['arbuz4'])

                bias_correction1 = 1 - arbuz2 ** state['arbuz6']
                bias_correction2 = 1 - arbuz3 ** state['arbuz6']
                step_size = group['arbuz1'] * math.sqrt(bias_correction2) / bias_correction1

                arbuz9.mul_(group['arbuz5']).add_(grad)

                p.data.addcdiv_(arbuz7, denom, value=-step_size)
                p.data.add_(arbuz9, alpha=-group['arbuz1'])

        return loss

torch.manual_seed(42)
X = torch.randn(100, 1)
y = 3 * X + 2 + torch.randn(100, 1) * 0.5

class LinearModel(nn.Module):
    def __init__(self):
        super(LinearModel, self).__init__()
        self.linear = nn.Linear(1, 1)

    def forward(self, x):
        return self.linear(x)

model = LinearModel()

criterion = nn.MSELoss()

optimizer = ADAM_i_eva(model.parameters(), arbuz1=0.01, arbuz2=0.9, arbuz3=0.999, arbuz4=1e-8, arbuz5=0.9)

num_epochs = 100
loss_values = []

for epoch in range(num_epochs):
    model.train()
    optimizer.zero_grad()
    outputs = model(X)
    loss = criterion(outputs, y)
    loss.backward()
    optimizer.step()

    loss_values.append(loss.item())
    if (epoch+1) % 10 == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

plt.plot(loss_values, label='Training Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()

model.eval()
with torch.no_grad():
    predictions = model(X)

plt.scatter(X.numpy(), y.numpy(), label='Тру Data')
plt.plot(X.numpy(), predictions.numpy(), color='red', label='Predicted sus')
plt.xlabel('X')
plt.ylabel('y')
plt.legend()
plt.show()
