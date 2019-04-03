import pandas as pd
import torch
import torch.nn as nn
import numpy as np
import phoneme_list as pl
from train import PhonDataset, collate_phon, PackedPhonModel, load_ckpt
from torch.utils.data import Dataset, DataLoader, TensorDataset
from ctcdecode import CTCBeamDecoder


# the p_map list
p_map = pl.PHONEME_MAP
p_map.append('%')
# print(len(p_map))
# print(p_map.index('%'))
# print(p_map)


# validation loader
val_label_path = './../data/wsj0_dev_merged_labels.npy'
val_label = np.load(val_label_path)

test_data_path = './../data/transformed_test_data.npy'
test_data = np.load(test_data_path, encoding='bytes')
test_label = val_label[:len(test_data)]

test_dataset = PhonDataset(test_data, test_label)
test_loader = DataLoader(test_dataset, shuffle=False, batch_size=1, collate_fn=collate_phon)


# load model
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# show string
decoder = CTCBeamDecoder(p_map, beam_width=100, blank_id=p_map.index('%'))

with torch.no_grad():
    path = './../result/exp4/id_13'
    model, _ = load_ckpt(path)
    model.to(DEVICE)

    total_dis = 0

    classification_result = []
    cnt = 0
    for inputs, targets in test_loader:
        if cnt % 50 == 0:
            print(cnt)

        output = model(inputs)

        output = torch.transpose(output, 0, 1)
        output = torch.exp(output)
        output_decode, _, _, out_seq_len = decoder.decode(output)

        pred = "".join(p_map[o] for o in output_decode[0, 0, :out_seq_len[0, 0]])

        line = (cnt, pred)
        classification_result.append(line)
        cnt += 1
        
class_df = pd.DataFrame(classification_result, columns=["Id", "Predicted"])
class_df.to_csv('./out.csv', index=False)

