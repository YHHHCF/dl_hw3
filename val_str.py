import torch
import torch.nn as nn
import numpy as np
import phoneme_list as pl
from train import PhonDataset, collate_phon, PackedPhonModel, load_ckpt
import Levenshtein as L
from torch.utils.data import Dataset, DataLoader, TensorDataset
from ctcdecode import CTCBeamDecoder


# the p_map list
p_map = pl.PHONEME_MAP
p_map.append('%')
# print(len(p_map))
# print(p_map.index('%'))
# print(p_map)


# validation loader
val_data_path = './../data/wsj0_dev.npy'
val_label_path = './../data/wsj0_dev_merged_labels.npy'
val_data = np.load(val_data_path, encoding='bytes')
val_label = np.load(val_label_path)
val_dataset = PhonDataset(val_data, val_label)
val_loader = DataLoader(val_dataset, shuffle=False, batch_size=1, collate_fn=collate_phon)


# load model
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# show string
decoder = CTCBeamDecoder(p_map, beam_width=100, blank_id=p_map.index('%'))

with torch.no_grad():
    path = './../result/exp4/id_14'
    model, _ = load_ckpt(path)
    model.to(DEVICE)

    total_dis = 0
    cnt = 0
    for inputs, targets in val_loader:
        cnt += 1

        if (cnt % 50 == 0):
            print(total_dis / cnt)
            print(cnt)

        output = model(inputs)

        output = torch.transpose(output, 0, 1)
        output = torch.exp(output)
        output_decode, _, _, out_seq_len = decoder.decode(output)

        for i in range(output.size(0)):
            pred = "".join(p_map[o] for o in output_decode[i, 0, :out_seq_len[i, 0]])

        true = "".join(p_map[o] for o in targets[0])

        dis = L.distance(pred, true)

        total_dis += dis

    total_dis /= cnt

    print("total_dis:", total_dis, idx)
    print("")
        

        