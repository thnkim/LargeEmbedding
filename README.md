# LargeEmbedding
Pytorch module for Large-scale word Embedding.

When you want to try embedding of a large number of words (for example, more than 10M+, 128D), it may not be easy since as-is Embedding layer tries to allocate one big tensor in one device. It will also try to secure additional space for the optimization. LargeEmbedding simply partition the LUT into multiple devices. 

Currently, a LUT will be evenly partitioned into pages of which size is [*page_size*], simply distributed into [*num_devices*] GPUs. It will be upgraded to be more flexible.

# How to use
```python
import torch
from torch.autograd import Variable
from largeEmbedding import LargeEmbedding

embedding = LargeEmbedding(n_words=50000, dim_word=4, page_size=10000, num_devices=2, use_cuda=True)

x = Variable(torch.LongTensor([[0, 1, 10000, 30100], [10000, 1, 30100, 0]]).cuda())
print('embedding:', embedding(x))

# for sanity check. the result above should be an arrangement of the following three tensors
print('test1:', embedding.embeddings[0](Variable(torch.LongTensor([0, 1]).cuda(0))))
print('test2:', embedding.embeddings[1](Variable(torch.LongTensor([0]).cuda(1))))
print('test3:', embedding.embeddings[3](Variable(torch.LongTensor([100]).cuda(1))))
```
