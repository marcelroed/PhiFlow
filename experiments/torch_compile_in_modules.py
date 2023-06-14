import os
os.environ['CUDA_VISIBLE_DEVICES'] = '5'
import torch
# How does TorchInductor work? What are the requirements for getting actually fast triton code?

DEV = torch.device('cuda:0')


class IndirectTensor:
    def __init__(self, torch_tensor):
        self.inner = torch_tensor

    def __add__(self, other):
        return IndirectTensor(self.inner + other.inner)

    def __sub__(self, other):
        return IndirectTensor(self.inner - other.inner)


def compiled_indirect():

    @torch.compile
    def adds_things(t1, t2):
        return t1 + t2

    t1 = IndirectTensor(torch.arange(10_000_000, device=DEV))
    t2 = IndirectTensor(torch.arange(10_000_000, 20_000_000, device=DEV))

    adds_things(t1, t2)


if __name__ == '__main__':
    import torch._inductor.config as ic
    ic.debug = True
    ic.trace.enabled = True
    compiled_indirect()