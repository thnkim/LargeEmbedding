# LargeEmbedding
Pytorch module for Large-scale word Embedding.

When you want to try embedding of a large number of words (for example, more than 10M+, 128D), it may not be easy since as-is Embedding layer tries to allocate one big tensor in one device. It will also try to secure additional space for the optimization. LargeEmbedding simply partition the LUT into multiple devices. 

Currently, we have two partitioning options:
1. LUT will be evenly partitioned into pages of which size is [*page_size*], simply distributed into [*num_devices*] GPUs. Simply, *page[i]* will be on the device *i % num_devices*.
2. LUT will be evenly partitioned into pages of which size is [*page_size*]. But each page can be distributed into specified device or devices. See examples below.

# How to use
```python
import torch
from torch.autograd import Variable
from largeEmbedding import LargeEmbedding

'''
50000 x 4 LUT will be partitioned into five 10000x4 LUT's. 
In this case, device_list is specified, thus num_devices will be ignored.
The first page will be data-parallelized by device 0 and 1.
The second page will be data-parallelized by device 0 and 3.
The third, fourth, and fifth pages will be on device 4, 5, and 6, respectively.
'''
embedding = LargeEmbedding(n_words=50000, dim_word=4, page_size=10000,
                           num_devices=-1, use_cuda=True,
                           device_list=[(0, 1), (0, 3), 4, 5, 6])

x = Variable(torch.LongTensor([[0, 1, 10000, 30100], [10000, 1, 30100, 0]]).cuda())
print('embedding:', embedding(x))

'''
Sanity check of the above embedding. 
Since the page size is 10000, word[10000] will be at the first element of the second page 
(= embedding.embeddings[1].weight[0]). It should be accessed by Variable(torch.LongTensor([0])).
Also since the page (embedding.embeddings[1]) is mirrored on device 0 and 3, the index variable 
(torch.LongTensor([0])) should also be on device 0. nn.DataParallel requests it to be on the first
device of the device_list, so it should be on 0, not 3.
word[30100] will be at index 100 of the 4th page (= embedding.embeddings[3].weight[100]).
'''
print('test1:', embedding.embeddings[0](Variable(torch.LongTensor([0, 1]).cuda(embedding.page_devices[0]))))
print('test2:', embedding.embeddings[1](Variable(torch.LongTensor([0]).cuda(embedding.page_devices[1]))))
print('test3:', embedding.embeddings[3](Variable(torch.LongTensor([100]).cuda(embedding.page_devices[3]))))

'''
Another sanity check. In this case, LUT is partitioned into one page, mirrored into 4 GPUs.
So the direct access of elements in a specific page is different from the above example.
'''
embedding = LargeEmbedding(n_words=50000, dim_word=4, page_size=50000,
                           num_devices=-1, use_cuda=True,
                           device_list=[(0, 1, 2, 3)])
x = Variable(torch.LongTensor([[0, 1, 10000, 30100], [10000, 1, 30100, 0]]).cuda())
print('embedding:', embedding(x))

print('test1:', embedding.embeddings[0](Variable(torch.LongTensor([0, 1]).cuda(embedding.page_devices[0]))))
print('test2:', embedding.embeddings[0](Variable(torch.LongTensor([10000]).cuda(embedding.page_devices[0]))))
print('test3:', embedding.embeddings[0](Variable(torch.LongTensor([30100]).cuda(embedding.page_devices[0]))))
```
