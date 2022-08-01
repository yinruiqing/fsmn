import torch
from torch import nn
import torch.nn.functional as F


class FSMN(nn.Module):
    def __init__(self, memory_size, input_size, output_size):
        super().__init__()
        self.memory_size = memory_size
        self.output_size = output_size
        self.input_size = input_size
        self._W1 = nn.Parameter(torch.Tensor(self.input_size, self.output_size))
        self._W2 = nn.Parameter(torch.Tensor(self.input_size, self.output_size))
        self._bias = nn.Parameter(torch.Tensor(self.output_size))
        self._memory_weights = nn.Parameter(torch.Tensor(self.memory_size))

        nn.init.xavier_uniform_(self._W1)
        nn.init.xavier_uniform_(self._W2)
        nn.init.ones_(self._bias)
        nn.init.xavier_uniform_(self._memory_weights)

    def get_memory_matrix(self, num_steps):
        memory_matrix = []
        for step in range(num_steps):
            left_num = max(0, step + 1 - self._memory_size)
            right_num = num_steps - step - 1
            mem = self._memory_weights[0:min(step, self.memory_size) + 1].flip(-1)
            d_batch = F.pad(mem, (left_num, right_num))
            memory_matrix.append(d_batch)
        memory_matrix = torch.stack((memory_matrix), 0)
        return memory_matrix

    def forward(self, input_data):
        num_steps = input_data.size(1)
        memory_matrix = self.get_memory_matrix(num_steps)
        h_hatt = torch.matmul(memory_matrix, input_data)
        h = torch.matmul(input_data, self._W1)
        h += torch.matmul(h_hatt, self._W2) + self._bias
        return h


class CSFSMN(FSMN):
    def __init__(self, memory_size, input_size, output_size):
        super().__init__(memory_size, input_size, output_size)
        self._W1 = nn.Parameter(torch.Tensor(self.input_size, self.output_size))
        self._W2 = nn.Parameter(torch.Tensor(self.input_size, self.output_size))
        self._bias1 = nn.Parameter(torch.Tensor(self.project_size))
        self._bias2 = nn.Parameter(torch.Tensor(self.output_size))
        self._memory_weights = nn.Parameter(torch.Tensor(self.memory_size))

        nn.init.xavier_uniform_(self._W1)
        nn.init.xavier_uniform_(self._W2)
        nn.init.ones_(self._bias1)
        nn.init.ones_(self._bias2)
        nn.init.xavier_uniform_(self._memory_weights)

    def forward(self, input_data):
        num_steps = input_data.size(1)
        p = torch.matmul(input_data, self._W1) + self._bias1
        memory_matrix = self.get_memory_matrix(num_steps)
        p = torch.matmul(memory_matrix, p)
        p = torch.matmul(p, self._W2) + self._bias2
        return p


class VFSMN(nn.Module):
    def __init__(self, memory_size, input_size, output_size):
        super().__init__()
        self.memory_size = memory_size
        self.output_size = output_size
        self.input_size = input_size
        self._W1 = nn.Parameter(torch.Tensor(self.input_size, self.output_size))
        self._W2 = nn.Parameter(torch.Tensor(self.input_size, self.output_size))
        self._bias = nn.Parameter(torch.Tensor(self.output_size))
        self._memory_weights = nn.Parameter(torch.Tensor(self.memory_size + 1, self.input_size))

        nn.init.xavier_uniform_(self._W1)
        nn.init.xavier_uniform_(self._W2)
        nn.init.ones_(self._bias)
        nn.init.xavier_uniform_(self._memory_weights)
        with torch.no_grad():
            self._memory_weights[0] = 0

    def forward(self, input_data):
        num_steps = input_data.size(1)
        memory_matrix = torch.ones((num_steps, num_steps), requires_grad=False).tril(-1).cumsum(0).triu(
            - self._memory_size).long()
        memory_matrix = memory_matrix.unsqueeze(0).expand(input_data.size(0), -1, -1)
        memory = self._memory_weights[memory_matrix].to(self.device)
        h_hatt = torch.einsum('bijd,bjd->bid', memory, input_data)  # 'bijd,bjd->bid'
        h = torch.matmul(input_data, self._W1)
        h += torch.matmul(h_hatt, self._W2) + self._bias
        return h
