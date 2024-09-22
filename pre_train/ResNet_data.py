
import numpy as np
import torch

class DealDataset(torch.utils.data.Dataset):
    def __init__(self, dataset, cuda):
        super(DealDataset).__init__()
        self.input_data = dataset[:, 0]
        self.label_data = dataset[:, 1]
        self.lens = dataset.shape[0]
        self.cuda = cuda

    def onehot_change(self, onehot_label):
        torch_label = [np.argmax(np.array(onehot_label))]

        return torch_label

    def __getitem__(self, index):

        x = self.input_data[index][20:180, 5:165, :3].swapaxes(0, 2).swapaxes(1, 2)
        label = self.onehot_change((self.label_data[index][:6]+self.label_data[index][7:]))
        if self.cuda:
            return torch.FloatTensor(x).cuda(), \
                   torch.squeeze(torch.LongTensor(label)).cuda()

        else:
            return torch.FloatTensor(x), \
                   torch.squeeze(torch.LongTensor(label))

    def __len__(self):
        return self.lens