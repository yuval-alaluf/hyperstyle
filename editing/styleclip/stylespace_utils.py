import torch

STYLESPACE_DIMENSIONS = [512 for _ in range(15)] + [256, 256, 256] + [128, 128, 128] + [64, 64, 64] + [32, 32]

TORGB_INDICES = list(range(1, len(STYLESPACE_DIMENSIONS), 3))
STYLESPACE_INDICES_WITHOUT_TORGB = [i for i in range(len(STYLESPACE_DIMENSIONS)) if i not in TORGB_INDICES][:11]

def features_channels_to_s(s_without_torgb, s_std):
    s = []
    start_index_features = 0
    for c in range(len(STYLESPACE_DIMENSIONS)):
        if c in STYLESPACE_INDICES_WITHOUT_TORGB:
            end_index_features = start_index_features + STYLESPACE_DIMENSIONS[c]
            s_i = s_without_torgb[start_index_features:end_index_features] * s_std[c]
            start_index_features = end_index_features
        else:
            s_i = torch.zeros(STYLESPACE_DIMENSIONS[c]).cuda()
        s_i = s_i.view(1, 1, -1, 1, 1)
        s.append(s_i)
    return s