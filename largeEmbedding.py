import numpy
import torch
import torch.nn as nn
from torch.autograd import Variable


class LargeEmbedding(nn.Module):
	def __init__(self, n_tracks, dim_track, page_size=0, use_cuda=True):
		super(LargeEmbedding, self).__init__()
		self.n_tracks = n_tracks
		self.dim_track = dim_track
		self.use_cuda = use_cuda

		if page_size <= 0:
			page_size = n_tracks

		self.page_size = page_size
		self.num_pages = (n_tracks + self.page_size - 1) // self.page_size
		print('making {0} LUTs'.format(self.num_pages))
		embedding_list = [nn.Embedding(self.page_size, self.dim_track, sparse=(not use_cuda)) for i in range(self.num_pages)]
		if self.use_cuda:
			for i, embedding in enumerate(embedding_list):
				embedding.cuda(i % 2)
		self.embeddings = nn.ModuleList(embedding_list)
		print(self.embeddings)

	def forward(self, indices_):
		indices = indices_.view(1, -1)

		y = torch.FloatTensor(1, indices.size(-1), self.dim_track)
		index_seq = torch.arange(0, indices.size(-1)).long().view(1, -1)
		if self.use_cuda:
			y = y.cuda()
			index_seq = index_seq.cuda()
		y = Variable(y)
		index_seq = Variable(index_seq, requires_grad=False)

		page_offset = 0
		for i in range(self.num_pages):
			mask_i = torch.min(torch.ge(indices, page_offset), torch.lt(indices, page_offset+self.page_size))
			#
			masked_idx_i = torch.masked_select(index_seq, mask_i)
			if masked_idx_i.dim() == 0:
				page_offset += self.page_size
				continue
			indices_i = torch.index_select(indices, 1, masked_idx_i) - page_offset
			if self.use_cuda:
				indices_i = indices_i.cuda(i % 2)
				v_i = self.embeddings[i](indices_i).cuda()
			else:
				v_i = self.embeddings[i](indices_i)
			y.index_copy_(1, masked_idx_i, v_i)
			#
			page_offset += self.page_size

		y = y.view(indices_.size(0), indices_.size(1), self.dim_track)

		return y


if __name__ == '__main__':
	embedding = LargeEmbedding(50000, 4, 10000, use_cuda=True)

	x = Variable(torch.LongTensor([[0, 1, 10000, 30100], [10000, 1, 30100, 0]]).cuda())
	print('embedding:', embedding(x))

	print('test1:', embedding.embeddings[0](Variable(torch.LongTensor([0, 1]).cuda(0))))
	print('test2:', embedding.embeddings[1](Variable(torch.LongTensor([0]).cuda(1))))
	print('test3:', embedding.embeddings[3](Variable(torch.LongTensor([100]).cuda(1))))

