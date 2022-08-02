import torch
from fsmn import FSMN, CSFSMN, VFSMN, VFSMNv2, CVFSMN, CVFSMNv2

batch_size = 4
input_size = 8
timestep = 16
output_size = 4
memory_size = 5
projection_size = 6
X = torch.randn(batch_size, timestep, input_size)


def test_FSMN():
    fsmn = FSMN(memory_size, input_size, output_size)
    print(fsmn(X).shape)


def test_CSFSMN():
    fsmn = CSFSMN(memory_size, input_size, output_size, projection_size)
    print(fsmn(X).shape)


def test_VFSMN():
    fsmn = VFSMN(memory_size, input_size, output_size)
    print(fsmn(X).shape)


def test_VFSMNv2():
    fsmn = VFSMNv2(memory_size, input_size, output_size)
    print(fsmn(X).shape)


def test_CVFSMN():
    fsmn = CVFSMN(memory_size, input_size, output_size, projection_size)
    print(fsmn(X).shape)


def test_CVFSMNv2():
    fsmn = CVFSMNv2(memory_size, input_size, output_size, projection_size)
    print(fsmn(X).shape)


if __name__ == '__main__':
    test_FSMN()
    test_CSFSMN()
    test_VFSMN()
    test_VFSMNv2()
    test_CVFSMN()
    test_CVFSMNv2()
