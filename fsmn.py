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
        nn.init.uniform_(self._memory_weights)

    def get_memory_matrix(self, num_steps):
        memory_matrix = []
        for step in range(num_steps):
            left_num = max(0, step + 1 - self.memory_size)
            right_num = num_steps - step - 1
            mem = self._memory_weights[0:min(step, self.memory_size) + 1].flip(-1)
            d_batch = F.pad(mem, (left_num, right_num))
            memory_matrix.append(d_batch)
        memory_matrix = torch.stack(memory_matrix, 0)
        return memory_matrix

    def forward(self, input_data):
        num_steps = input_data.size(1)
        memory_matrix = self.get_memory_matrix(num_steps)
        h_hatt = torch.matmul(memory_matrix, input_data)
        h = torch.matmul(input_data, self._W1)
        h += torch.matmul(h_hatt, self._W2) + self._bias
        return h


class CSFSMN(FSMN):
    def __init__(self, memory_size, input_size, output_size, projection_size):
        super().__init__(memory_size, input_size, output_size)
        self.projection_size = projection_size
        self._W1 = nn.Parameter(torch.Tensor(self.input_size, self.projection_size))
        self._W2 = nn.Parameter(torch.Tensor(self.projection_size, self.output_size))
        self._bias1 = nn.Parameter(torch.Tensor(self.projection_size))
        self._bias2 = nn.Parameter(torch.Tensor(self.output_size))
        self._memory_weights = nn.Parameter(torch.Tensor(self.memory_size))

        nn.init.xavier_uniform_(self._W1)
        nn.init.xavier_uniform_(self._W2)
        nn.init.ones_(self._bias1)
        nn.init.ones_(self._bias2)
        nn.init.uniform_(self._memory_weights)

    def forward(self, input_data):
        num_steps = input_data.size(1)
        memory_matrix = self.get_memory_matrix(num_steps)
        p = torch.matmul(input_data, self._W1) + self._bias1
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
        memory_matrix = torch.ones((num_steps, num_steps), requires_grad=False).tril(0).cumsum(0).triu(
            - self.memory_size + 1).long()
        memory_matrix = memory_matrix.unsqueeze(0).expand(input_data.size(0), -1, -1)
        memory = self._memory_weights[memory_matrix]
        h_hatt = torch.einsum('bijd,bjd->bid', memory, input_data)  # 'bijd,bjd->bid'
        h = torch.matmul(input_data, self._W1)
        h += torch.matmul(h_hatt, self._W2) + self._bias
        return h


class VFSMNv2(nn.Module):
    def __init__(self, memory_size, input_size, output_size):
        super().__init__()
        self.memory_size = memory_size
        self.output_size = output_size
        self.input_size = input_size
        self._W1 = nn.Parameter(torch.Tensor(self.input_size, self.output_size))
        self._W2 = nn.Parameter(torch.Tensor(self.input_size, self.output_size))
        self._bias = nn.Parameter(torch.Tensor(self.output_size))
        self._memory_weights = nn.Parameter(torch.Tensor(self.input_size, 1, self.memory_size))

        nn.init.xavier_uniform_(self._W1)
        nn.init.xavier_uniform_(self._W2)
        nn.init.ones_(self._bias)
        nn.init.xavier_uniform_(self._memory_weights)

    def forward(self, input_data):
        input_data_T = F.pad(input_data.transpose(1, 2), (self.memory_size - 1, 0))
        h_hatt = F.conv1d(input_data_T, self._memory_weights, groups=self.input_size)
        h_hatt = h_hatt.transpose(1, 2)
        h = torch.matmul(input_data, self._W1)
        h += torch.matmul(h_hatt, self._W2) + self._bias
        return h


class CVFSMN(nn.Module):
    def __init__(self, memory_size, input_size, output_size, projection_size):
        super().__init__()
        self.memory_size = memory_size
        self.output_size = output_size
        self.input_size = input_size
        self.projection_size = projection_size
        self._W1 = nn.Parameter(torch.Tensor(self.input_size, self.projection_size))
        self._W2 = nn.Parameter(torch.Tensor(self.projection_size, self.output_size))
        self._bias1 = nn.Parameter(torch.Tensor(self.projection_size))
        self._bias2 = nn.Parameter(torch.Tensor(self.output_size))
        self._memory_weights = nn.Parameter(torch.Tensor(self.memory_size + 1, self.projection_size))

        nn.init.xavier_uniform_(self._W1)
        nn.init.xavier_uniform_(self._W2)
        nn.init.ones_(self._bias1)
        nn.init.ones_(self._bias2)
        nn.init.xavier_uniform_(self._memory_weights)
        with torch.no_grad():
            self._memory_weights[0] = 0

    def forward(self, input_data):
        num_steps = input_data.size(1)
        memory_matrix = torch.ones((num_steps, num_steps), requires_grad=False).tril(0).cumsum(0).triu(
            - self.memory_size + 1).long()
        memory_matrix = memory_matrix.unsqueeze(0).expand(input_data.size(0), -1, -1)
        memory = self._memory_weights[memory_matrix]
        p = torch.matmul(input_data, self._W1) + self._bias1
        p = torch.einsum('bijd,bjd->bid', memory, p)
        p = torch.matmul(p, self._W2) + self._bias2
        return p


class CVFSMNv2(nn.Module):
    def __init__(self, memory_size, input_size, output_size, projection_size):
        super().__init__()
        self.memory_size = memory_size
        self.output_size = output_size
        self.input_size = input_size
        self.projection_size = projection_size
        self._W1 = nn.Parameter(torch.Tensor(self.input_size, self.projection_size))
        self._W2 = nn.Parameter(torch.Tensor(self.projection_size, self.output_size))
        self._bias1 = nn.Parameter(torch.Tensor(self.projection_size))
        self._bias2 = nn.Parameter(torch.Tensor(self.output_size))
        self._memory_weights = nn.Parameter(torch.Tensor(self.projection_size, 1, self.memory_size))

        nn.init.xavier_uniform_(self._W1)
        nn.init.xavier_uniform_(self._W2)
        nn.init.ones_(self._bias1)
        nn.init.ones_(self._bias2)
        nn.init.xavier_uniform_(self._memory_weights)

    def forward(self, input_data):
        p = torch.matmul(input_data, self._W1) + self._bias1
        p_T = F.pad(p.transpose(1, 2), (self.memory_size - 1, 0))
        p = F.conv1d(p_T, self._memory_weights, groups=self.projection_size).transpose(1, 2)
        p = torch.matmul(p, self._W2) + self._bias2
        return p
