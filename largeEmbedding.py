import numpy
import torch
import torch.nn as nn
from torch.autograd import Variable


class LargeEmbedding(nn.Module):
    def __init__(self, n_words, dim_word, page_size=0, num_devices=2, use_cuda=True, device_list=None):
        """
        :param n_words: # of words to embed (= size of the dictionary of embeddings) 
        :param dim_word: dimension of a word's embedding (= the size of each embedding vector)
        :param page_size: the size of a page, that is, # of words in a partition from n_words x dim_words LUT.
                           For example, if n_words = 1000000, dim_word = 32, and page_size = 100000, 
                           a LUT of 1000000x32 is partitioned into ten pages each of which is 100000x32 LUTs. 
                           If n_words is not a multiple of page_size, size of the last page will be smaller.
        :param num_devices: # of GPU devices that will keep the pages. It will be ignored when device_list is provided.
        :param use_cuda: It should be True at this time. 
        :param device_list: a list GPU devices that will keep the pages. 
                             device_list[i] will be an integer (device-id) or a tuple (device-ids) to keep i-th page.
                             If a tuple is specified for a page, the page will be wrapped by torch.nn.DataParallel,
                             but its efficiency has not been tested yet. 
                             So far, to use 8 GPUs, it seems partitioning 10M words of LUT into 8 pages each of which 
                             size is 1.25M words is faster than mirroring 10M words of LUTs on 8 GPUs and doing 
                             data-parallelizing, at least for the training. 
        """
        super(LargeEmbedding, self).__init__()
        self.n_words = n_words
        self.dim_word = dim_word
        self.num_devices = num_devices
        self.use_cuda = use_cuda

        if page_size <= 0:
            page_size = n_words

        self.page_size = page_size
        self.num_pages = (n_words + self.page_size - 1) // self.page_size
        print('making {0} LUTs'.format(self.num_pages))
        assert (device_list is None) or (self.num_pages == len(device_list)), \
            'len(device_list) should be the same as # of LUTs (={0})'.format(self.num_pages)

        self.page_devices = []
        embedding_list = [nn.Embedding(self.page_size, self.dim_word, sparse=(not use_cuda)) for i in
                          range(self.num_pages)]
        if self.use_cuda:
            for i, embedding in enumerate(embedding_list):
                if device_list is not None:
                    device = device_list[i]
                    if (type(device) is list) or (type(device) is tuple):
                        embedding_list[i] = nn.DataParallel(embedding, device_ids=device).cuda()
                        self.page_devices.append(device[0])
                    else:
                        embedding.cuda(device)
                        self.page_devices.append(device)
                else:
                    device = i % self.num_devices
                    embedding.cuda(device)
                    self.page_devices.append(device)
        else:
            pass
        #
        self.embeddings = nn.ModuleList(embedding_list)
        print(self.embeddings)

    def forward(self, indices_):
        indices = indices_.view(1, -1)

        y = torch.FloatTensor(1, indices.size(-1), self.dim_word)
        index_seq = torch.arange(0, indices.size(-1)).long().view(1, -1)
        if self.use_cuda:
            y = y.cuda()
            index_seq = index_seq.cuda()
        y = Variable(y)
        index_seq = Variable(index_seq, requires_grad=False)

        page_offset = 0
        for i in range(self.num_pages):
            mask_i = torch.min(torch.ge(indices, page_offset), torch.lt(indices, page_offset + self.page_size))
            #
            masked_idx_i = torch.masked_select(index_seq, mask_i)
            if masked_idx_i.dim() == 0:
                page_offset += self.page_size
                continue
            indices_i = torch.index_select(indices, 1, masked_idx_i) - page_offset
            if self.use_cuda:
                indices_i = indices_i.cuda(self.page_devices[i])
                try:
                    v_i = self.embeddings[i](indices_i)
                    v_i = v_i.cuda()
                except:
                    print(indices_i, page_offset)
                    print(self.page_devices[i])
                    print(self.embeddings[i])
                    print(self.embeddings[i].device_ids)
                    print(indices_i.get_device())
                    print(self.embeddings[i](indices_i.cuda(2)))
            else:
                v_i = self.embeddings[i](indices_i)
            y.index_copy_(1, masked_idx_i, v_i)
            #
            page_offset += self.page_size

        y = y.view(indices_.size(0), indices_.size(1), self.dim_word)

        return y


if __name__ == '__main__':
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
