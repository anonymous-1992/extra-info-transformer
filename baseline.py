import torch.nn as nn
import torch
import torch.nn.functional as F
import numpy as np
import random

np.random.seed(21)
random.seed(21)

class Lstnet(nn.Module):
    def __init__(self, hidRNN, hidCNN, hidSkip, CNN_kernel, skip,
                 seq_len, seq_len_pred, input_size, dr, device):
        super(Lstnet, self).__init__()
        self.device = device
        self.P = seq_len
        self.m = hidCNN
        self.hidR = hidRNN
        self.hidC = hidCNN
        self.hidS = hidSkip
        self.Ck = CNN_kernel
        self.skip = skip
        self.pt = int ((self.P - self.Ck) / self.skip)
        self.conv1 = nn.Conv2d(1, self.hidC, kernel_size=(self.Ck, self.m))
        self.GRU1 = nn.GRU(self.hidC, self.hidR)
        self.dropout = nn.Dropout(dr)
        self.proj = nn.Linear(input_size, self.hidC)
        if self.skip > 0:
            self.GRUskip = nn.GRU(self.hidC, self.hidS)
            self.linear1 = nn.Linear(self.hidR + self.skip * self.hidS, seq_len_pred)
        else:
            self.linear1 = nn.Linear(self.hidR, seq_len_pred)
        self.linear2 = nn.Linear(seq_len_pred, seq_len_pred)

    def forward(self, x):
        batch_size = x.size(0)
        x = self.proj(x)

        # CNN
        c = x.view(-1, 1, self.P, self.m)
        c = F.relu(self.conv1(c))
        c = self.dropout(c)
        c = torch.squeeze(c, 3)

        # RNN
        r = c.permute(2, 0, 1).contiguous()
        _, r = self.GRU1(r)
        r = self.dropout(r.squeeze(0))

        if self.skip > 0:
            s = c[:, :, int(-self.pt * self.skip):].contiguous()
            s = s.view(batch_size, self.hidC, self.pt, self.skip)
            s = s.permute(2, 0, 3, 1).contiguous()
            s = s.view(self.pt, batch_size * self.skip, self.hidC)
            _, s = self.GRUskip(s)
            s = s.view(batch_size, self.skip * self.hidS)
            s = self.dropout(s)
            r = torch.cat((r, s), 1)

        r = r.unsqueeze(-1)
        res = self.linear1(r.permute(0, 2, 1))
        res = self.linear2(res).permute(0, 2, 1)
        res = torch.sigmoid(res)
        return res


class RNConv(nn.Module):

    def __init__(self,input_size, output_size,
                 out_channel, kernel, n_layers,
                 hidden_size, seq_len, seq_pred_len,
                 device, d_r):

        super(RNConv, self).__init__()
        self.n_layers = n_layers
        self.hidden_size = hidden_size
        self.conv = nn.Conv1d(input_size, hidden_size, kernel)
        self.lstm = nn.LSTM(hidden_size, hidden_size, n_layers, dropout=d_r)
        self.dropout1 = nn.Dropout(d_r)
        self.linear2 = nn.Linear(hidden_size, output_size)
        self.proj_out = nn.Linear(seq_len, seq_pred_len, bias=False)
        self.kernel_size = kernel
        self.dilation = 1
        self.device = device

    def forward(self, x, hidden=None):

        seq_len, b, f = x.shape
        x = x.view(b, f, seq_len)

        x = x.contiguous().view(b, f, seq_len)
        padding = (self.kernel_size - 1) * self.dilation
        x = F.pad(x, (padding, 0))

        x_out = self.conv(x)

        x_en_out = x_out.view(seq_len, b, -1)

        if hidden is None:
            hidden = torch.zeros(self.n_layers, b, self.hidden_size).to(self.device)

        output, _ = self.lstm(x_en_out, (hidden, hidden))
        output = output.permute(0, 2, 1)

        outputs = self.proj_out(output)
        outputs = self.linear2(outputs.permute(0, 2, 1))

        return outputs


class RNN(nn.Module):
    def __init__(self, n_layers, hidden_size,
                 src_input_size, tgt_input_size,
                 rnn_type, device, d_r, seed):

        super(RNN, self).__init__()
        torch.manual_seed(seed)
        self.enc_lstm = nn.LSTM(hidden_size, hidden_size, n_layers, dropout=d_r)
        self.dec_lstm = nn.LSTM(hidden_size, hidden_size, n_layers, dropout=d_r)
        self.n_layers = n_layers
        self.hidden_size = hidden_size
        self.rnn_type = rnn_type
        self.linear_enc = nn.Linear(src_input_size, hidden_size, bias=False)
        self.linear_dec = nn.Linear(tgt_input_size, hidden_size, bias=False)
        self.linear2 = nn.Linear(hidden_size, 1, bias=False)
        self.device = device

    def forward(self, x_en, x_de, hidden=None):

        x_en = self.linear_enc(x_en).permute(1, 0, 2)
        x_de = self.linear_dec(x_de).permute(1, 0, 2)

        if hidden is None:
            hidden = torch.zeros(self.n_layers, x_en.shape[1], self.hidden_size).to(self.device)

        _, (hidden, state) = self.enc_lstm(x_en, (hidden, hidden))
        dec_output, _ = self.dec_lstm(x_de, (hidden, hidden))

        outputs = self.linear2(dec_output).transpose(0, 1)

        return outputs


class MLP(nn.Module):
    def __init__(self, n_layers, hidden_size,
                 input_size, output_size, seq_len_pred, device, dr):
        super(MLP, self).__init__()

        self.l1 = nn.Linear(input_size, output_size)
        self.conv1 = nn.Conv1d(input_size, hidden_size, kernel_size=1)
        self.conv2 = nn.Conv1d(hidden_size, output_size, kernel_size=1)
        self.l2 = nn.Linear(seq_len_pred, seq_len_pred)
        self.seq_len_pred = seq_len_pred
        self.dropout = nn.Dropout(dr)

        self.layer_norm = nn.LayerNorm(output_size)
        self.n_layers = n_layers
        self.device = device

    def forward(self, inputs):

        for _ in range(self.n_layers):
            residual = self.l1(inputs)
            output_1 = self.conv1(inputs.transpose(1, 2))
            output_1 = nn.ReLU()(output_1)
            output_1 = self.conv2(output_1).transpose(1, 2)
            output_1 = self.dropout(output_1)
        output = output_1 + residual

        output = self.l2(nn.Linear(inputs.size(1), self.seq_len_pred).
                         to(self.device)(output.permute(0, 2, 1)))
        return self.layer_norm(output.permute(0, 2, 1))


class CNN(nn.Module):
    def __init__(self, input_size, output_size,
                 out_channel, kernel, n_layers,
                 seq_len, seq_pred_len, device, d_r):
        super(CNN, self).__init__()
        self.conv = [nn.Conv1d(input_size, out_channel, kernel).to(device) for _ in range(n_layers)]
        self.dropout1 = [nn.Dropout(d_r) for _ in range(n_layers)]
        self.linear = nn.Linear(out_channel, output_size)
        self.n_layers = n_layers
        self.out_channel = out_channel
        self.proj = nn.Linear(out_channel, input_size)
        self.proj_out = nn.Linear(seq_len, seq_pred_len)
        self.kernel_size = kernel
        self.dilation = 1
        self.device = device

    def forward(self, x_en, x_de, training=True):

        de_seq_len = x_de.shape[1]
        b, seq_len, f = x_en.shape
        x_en = x_en.contiguous().view(b, f, seq_len)
        x_de = x_de.contiguous().view(b, f, de_seq_len)
        padding = (self.kernel_size - 1) * self.dilation
        x_en = F.pad(x_en, (padding, 0))
        x_de = F.pad(x_de, (padding, 0))
        proj2 = nn.Linear(seq_len+de_seq_len, de_seq_len).to(self.device)

        for i in range(self.n_layers):
            x_en_out = self.conv[i](x_en)
            x_en_out = self.dropout1[i](x_en_out)

        x_en_out = self.proj(x_en_out.permute(0, 2, 1))
        x_en_out = x_en_out.permute(0, 2, 1)
        x_de = torch.cat((x_en_out, x_de), dim=2)

        for i in range(self.n_layers):
            x_de_out = self.conv[i](x_de)
            x_de_out = self.dropout1[i](x_de_out)

        x_de_out = proj2(x_de_out)
        x_de_out = self.proj_out(x_de_out)
        output = self.linear(x_de_out.contiguous().view(b, -1, self.out_channel))

        return output