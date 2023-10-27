from hw_asr.base import BaseModel
import torch
from torch import nn

class BatchRNN(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, rnn_type: str):
        super().__init__()
        self.rnn = getattr(nn, rnn_type)(
            input_size=input_size,
            hidden_size=hidden_size,
            bias = True,
            bidirectional=True,
            batch_first=True,
        )
        self.bn = nn.BatchNorm1d(hidden_size)

    def forward(self, x: torch.Tensor):
        x, _ = self.rnn(x.transpose(1, 2).contiguous())
        x = x.view(x.size(0), x.size(1), 2, -1).sum(-2).transpose(1, 2).contiguous()
        x = self.bn(x)
        return x


class DeepSpeech2(BaseModel):
    def __init__(self, n_feats: int, n_class: int, hidden_size: int, rnn_type: str, num_rnn_layers: int, num_conv_layers: int, **batch):
        super().__init__(n_feats, n_class, **batch)

        conv_layers = [
            nn.Conv2d(in_channels=1, out_channels=32, kernel_size=(41, 11), stride=(2, 2), padding=(20, 5)),
            nn.BatchNorm2d(num_features=32),
            nn.Hardtanh(0, 20, inplace=True),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(21, 11), stride=(2, 2), padding=(10, 5)),
            nn.BatchNorm2d(num_features=32),
            nn.Hardtanh(0, 20, inplace=True),
            nn.Conv2d(in_channels=32, out_channels=96, kernel_size=(21, 11), stride=(2, 1), padding=(10, 5)),
            nn.BatchNorm2d(num_features=96),
            nn.Hardtanh(0, 20, inplace=True), 
        ]

        self.conv = nn.Sequential(*(conv_layers[:num_conv_layers * 3]))
        self.last_conv  = conv_layers[num_conv_layers * 3 - 3]

        conv_output_size = n_feats
        for i in range(num_conv_layers * 3):
            if isinstance(self.conv[i], nn.Conv2d):
                conv_output_size = ((conv_output_size + 2 * self.conv[i].padding[0] - self.conv[i].kernel_size[0]) // self.conv[i].stride[0] + 1)
        
        conv_output_size *= (96 if num_conv_layers == 3 else 32)
        
        rnn_layers = [BatchRNN(conv_output_size if i == 0 else hidden_size, hidden_size, rnn_type) for i in range(num_rnn_layers)]
        self.rnns = nn.Sequential(*rnn_layers)
        self.head = nn.Linear(hidden_size, n_class)
    
    def forward(self, spectrogram, **batch):
        x = spectrogram.unsqueeze(1)
        x = self.conv(x)
        sizes = x.size()
        x = x.view(sizes[0], sizes[1] * sizes[2], sizes[3])
        x = self.rnns(x).transpose(1, 2).contiguous()
        x = self.head(x)
        return x
    
    def transform_input_lengths(self, input_lengths: torch.Tensor):
        max_len = input_lengths.max().item()
        for i in range(len(self.conv)):
            if isinstance(self.conv[i], nn.Conv2d):
                max_len = ((max_len + 2 * self.conv[i].padding[1] - self.conv[i].kernel_size[1]) // self.conv[i].stride[1] + 1)
        return torch.full((input_lengths.size(0),), max_len, dtype=torch.long)




