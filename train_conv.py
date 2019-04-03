import phoneme_list as pl
import numpy as np
import torch
import torch.nn as nn
import torch.nn.utils.rnn as rnn

import Levenshtein as L
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, TensorDataset
from ctcdecode import CTCBeamDecoder
from tensorboardX import SummaryWriter

# phonome dataset, init dataset with data and label
class PhonDataset(Dataset):
    def __init__(self, phon_list, labels):
        self.data = [torch.tensor(phon) for phon in phon_list]
        self.labels = [torch.tensor(label) for label in labels]

    def __getitem__(self, i):
        DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
        data = self.data[i]
        label = self.labels[i]
        data = data.type(torch.float32)
#         augmentation = torch.normal(mean=torch.zeros(data.shape[0], data.shape[1]), std=0.1)
#         data += augmentation
        return data.to(DEVICE), label.to(DEVICE)

    def __len__(self):
        return len(self.labels)


# collate_phon return your data sorted by length
def collate_phon(phon_list):
    inputs, targets = zip(*phon_list)
    lens = [len(phon) for phon in inputs]
    seq_order = sorted(range(len(lens)), key=lens.__getitem__, reverse=True)
    inputs = [inputs[i] for i in seq_order]
    targets = [targets[i] for i in seq_order]
    return inputs, targets


def print_model(model):
    params = model.state_dict()
    keys = params.keys()
    for key in keys:
        print(key + ": ")
        print(torch.max(params[key]))
    return


# phonome dataloader
# Model that takes packed sequences in training
class PackedPhonModel(nn.Module):
    def __init__(self, in_size, hidden_size, out_size, nlayers):
        super(PackedPhonModel, self).__init__()
        self.in_size = in_size
        self.hidden_size = hidden_size
        self.nlayers = nlayers
        self.conv = nn.Conv1d(in_channels=in_size, out_channels=128, kernel_size=31, padding=15)
        self.rnn = nn.LSTM(input_size=128, hidden_size=hidden_size, num_layers=nlayers, bidirectional=True)
#         self.dropout = nn.Dropout(0.5)
        self.scoring1 = nn.Linear(2 * hidden_size, out_size)
        self.lsm = nn.LogSoftmax(dim=2)

    def forward(self, phon_list):  # list
        # go through conv layer        
        cnn_input = []
        for phon in phon_list:
#             print("phon shape", phon.shape)
            phon = torch.transpose(phon, 0, 1)
            phon = torch.reshape(phon, (1, phon.shape[0], phon.shape[1]))
            cnn_phon = self.conv(phon)
            cnn_phon = torch.reshape(cnn_phon, (cnn_phon.shape[1], cnn_phon.shape[2]))
            cnn_phon = torch.transpose(cnn_phon, 0, 1)
#             print("cnn_phon shape", cnn_phon.shape)
            cnn_input.append(cnn_phon)
        
        # pack and split the length sorted input into small pieces
        packed_input = rnn.pack_sequence(cnn_input)
#         packed_input = rnn.pack_sequence(phon_list)  # packed version

        hidden = None
        output_packed, hidden = self.rnn(packed_input, hidden)

        # get the output with dim 0 corresponding to packed_input
        output_padded, _ = rnn.pad_packed_sequence(output_packed)  # unpacked output (padded)
#         output_padded = self.dropout(output_padded)
        scores_flatten = self.scoring1(output_padded)  # concatenated logits
        scores_flatten = self.lsm(scores_flatten)

        return scores_flatten  # return concatenated logits
    
    
def save_ckpt(model, optimizer, val_loss, idx):
    id_name = 'id_' + str(idx)
    path = './../result/' + id_name

    torch.save({
        'exp_id': idx,
        'val_loss': val_loss,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }, path)
    
    return path

def load_ckpt(path):
    new_model = PackedPhonModel(40, 256, 47, 4)
    pretrained_ckpt = torch.load(path)
    new_model.load_state_dict(pretrained_ckpt['model_state_dict'])
    
    new_optimizer = torch.optim.Adam(new_model.parameters(), lr=0.001, weight_decay=1e-6)
    new_optimizer.load_state_dict(pretrained_ckpt['optimizer_state_dict'])

    for state in new_optimizer.state.values():
        for k, v in state.items():
            if isinstance(v, torch.Tensor):
                state[k] = v.cuda()
    return new_model, new_optimizer


def train(epochs, train_loader, val_loader, model, writer):
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    model.train()
    idx = 0
    for e in range(epochs):
        print("begin epoch: ", e)
        for inputs, targets in train_loader:
            idx += 1
            # inputs is a list of 64 frames, each frame is K * 40, with K varies
            # targets is a list of 64 target vectors, each containing T values, T varies
            in_lens = [len(phon) for phon in inputs]
            tar_lens = [len(tar) for tar in targets]

            in_lens = torch.tensor(in_lens)
            tar_lens = torch.tensor(tar_lens)

            packed_targets = torch.cat(targets, dim=0)

            outputs = model(inputs)
            loss = criterion(outputs, packed_targets, in_lens, tar_lens)
            
            writer.add_scalar('train/loss', loss.item(), idx)

            # perform backprop
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
        val_loss = val(model, val_loader, writer, e)
        model.train()
        save_ckpt(model, optimizer, val_loss, e)
            
            
# validation
def val(model, val_loader, writer, ep):
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    model.eval()
    
    with torch.no_grad():
        cnt = 0
        avg_loss = 0
        for inputs, targets in val_loader:
            cnt += 1
            in_lens = [len(phon) for phon in inputs]
            tar_lens = [len(tar) for tar in targets]

            in_lens = torch.tensor(in_lens)
            tar_lens = torch.tensor(tar_lens)

            packed_targets = torch.cat(targets, dim=0)

            outputs = model(inputs)
            loss = criterion(outputs, packed_targets, in_lens, tar_lens)
            avg_loss += loss.item()
        avg_loss /= cnt
        writer.add_scalar('val/loss', avg_loss, ep)
        return avg_loss
    
    
if __name__ == '__main__':
    n_stat = pl.N_STATES
    n_phon = pl.N_PHONEMES
    p_list = pl.PHONEME_LIST
    p_map = pl.PHONEME_MAP
    
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    train_data_path = './../data/wsj0_train.npy'
    train_label_path = './../data/wsj0_train_merged_labels.npy'

    val_data_path = './../data/wsj0_dev.npy'
    val_label_path = './../data/wsj0_dev_merged_labels.npy'

    test_path = './../data/transformed_test_data.npy'

    train_data = np.load(train_data_path, encoding='bytes')
    train_label = np.load(train_label_path)

    val_data = np.load(val_data_path, encoding='bytes')
    val_label = np.load(val_label_path)

    model = PackedPhonModel(40, 256, 47, 4)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-6)
#     optimizer = torch.optim.Adadelta(model.parameters(), lr=1.0, rho=0.9, eps=1e-06, weight_decay=0)

#     path = './../result/exp3/id_15'
#     model, optimizer = load_ckpt(path)

    train_dataset = PhonDataset(train_data, train_label)
    train_loader = DataLoader(train_dataset, shuffle=True, batch_size=64, collate_fn=collate_phon)

    val_dataset = PhonDataset(val_data, val_label)
    val_loader = DataLoader(val_dataset, shuffle=False, batch_size=64, collate_fn=collate_phon)

    criterion = nn.CTCLoss(blank=46)
    criterion = criterion.to(DEVICE)
    model = model.to(DEVICE)

    epochs = 20
    writer = SummaryWriter()

    train(epochs, train_loader, val_loader, model, writer)
    