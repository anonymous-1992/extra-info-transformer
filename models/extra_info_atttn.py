import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import math
import random

np.random.seed(1234)
random.seed(1234)


def get_attn_subsequent_mask(seq):
    attn_shape = [seq.size(0), seq.size(1), seq.size(1)]
    subsequent_mask = np.triu(np.ones(attn_shape), k=1)
    subsequent_mask = torch.from_numpy(subsequent_mask).int()
    return subsequent_mask


class PositionalEncoding(nn.Module):
    """Positional encoding."""
    def __init__(self, d_hid, device, max_len=1000):
        super(PositionalEncoding, self).__init__()
        # Create a long enough `P`
        self.P = torch.zeros((1, max_len, d_hid)).to(device)
        X = torch.arange(max_len, dtype=torch.float32).reshape(
            -1, 1) / torch.pow(10000, torch.arange(
            0, d_hid, 2, dtype=torch.float32) / d_hid)
        self.P[:, :, 0::2] = torch.sin(X)
        self.P[:, :, 1::2] = torch.cos(X)

    def forward(self, X):
        X = X + self.P[:, :X.shape[1], :].to(X.device)
        return X


class PositionalEncoding_2d(nn.Module):
    """Positional encoding."""
    def __init__(self, d_hid, device, max_len=500):
        super(PositionalEncoding_2d, self).__init__()
        # Create a long enough `P`
        self.P = torch.zeros((1, 1, max_len, d_hid)).to(device)
        X = torch.arange(max_len, dtype=torch.float32).reshape(
            -1, 1) / torch.pow(10000, torch.arange(
            0, d_hid, 2, dtype=torch.float32) / d_hid)
        self.P[:, :, :, 1::2] = torch.sin(X)
        self.P[:, :, :, 1::2] = torch.cos(X)

    def forward(self, X):
        X = X + self.P[:, :, :X.shape[2], :]
        return X


class ScaledDotProductAttention(nn.Module):

    def __init__(self, d_k, device, n_heads, l_k, b_size, n_ext_info,
                 kernel_s, kernel_b, attn_type, enc_attn=False):

        super(ScaledDotProductAttention, self).__init__()
        self.device = device
        self.d_k = d_k
        self.softmax = nn.Softmax(dim=-1)
        self.attn_type = attn_type
        self.enc_attn = enc_attn
        self.n_ext_info = n_ext_info
        self.Er = nn.Parameter(torch.randn(l_k, d_k).to(device), requires_grad=True)
        if "extra_info_attn" in self.attn_type:

            self.num_past_info = math.ceil(math.log2(b_size))
            padding_s = int((kernel_s - 1) / 2)
            padding_b = int((kernel_b - 1) / 2)
            stride_s = 1 if kernel_s == 1 else int(kernel_s / 2)
            stride_b = 1 if kernel_b == 1 else int(kernel_b / 2)
            l_k = math.floor(((l_k + 2 * padding_s - (kernel_s - 1) - 1) / stride_s) + 1)
            n_ext_info = math.floor(((n_ext_info + 2 * padding_b - kernel_b) / stride_b) + 1)
            kernel_max_pool_s = math.ceil(l_k / self.num_past_info)
            kernel_max_pool_b = math.ceil(n_ext_info / self.num_past_info)
            padding_max_pooling_s = int((kernel_max_pool_s - 1)/2)
            padding_max_pooling_b = int((kernel_max_pool_b - 1) / 2)
            self.m = math.floor(((l_k + 2 * padding_max_pooling_s - 1 *
                                  (kernel_max_pool_s - 1) - 1) / kernel_max_pool_s) + 1)
            self.n = math.floor(((n_ext_info + 2 * padding_max_pooling_b - 1 *
                                  (kernel_max_pool_b - 1) - 1) / kernel_max_pool_b) + 1)

            if "2d" in self.attn_type:
                self.conv2d = nn.Conv2d(in_channels=d_k*n_heads,
                                        out_channels=d_k*n_heads,
                                        kernel_size=(kernel_s, kernel_b),
                                        stride=(stride_s, stride_b),
                                        padding=(padding_s, padding_b)).to(device)

                self.max_pooling = \
                    nn.MaxPool2d(kernel_size=(kernel_max_pool_s, kernel_max_pool_b),
                                 padding=(padding_max_pooling_s, padding_max_pooling_b))
            else:
                self.conv2d = nn.Conv2d(in_channels=d_k*n_heads,
                                        out_channels=d_k*n_heads,
                                        kernel_size=(kernel_s, 1),
                                        stride=(stride_s, 1),
                                        padding=(padding_s, 0)).to(device)
                self.max_pooling = \
                    nn.MaxPool2d(kernel_size=(kernel_max_pool_s, kernel_max_pool_b),
                                 padding=(padding_max_pooling_s, padding_max_pooling_b))

            self.linear_k = nn.Parameter(torch.randn(self.n*self.m).to(device), requires_grad=True)

    def get_new_rep(self, tnsr):

        b, h, l_k, d = tnsr.shape
        tnsr = tnsr.reshape(l_k, h * d, b)
        tnsr = F.pad(tnsr, pad=(self.n_ext_info - 1, 0, 0, 0))
        tnsr = tnsr.unfold(-1, self.n_ext_info, 1)
        tnsr = tnsr.reshape(b, h * d, l_k, self.n_ext_info)
        tnsr = self.conv2d(tnsr)
        tnsr = self.max_pooling(tnsr)
        tnsr = tnsr.view(b, h, self.n*self.m, d)
        return tnsr

    def skew(self, QEr):
        # QEr.shape = (batch_size, num_heads, seq_len, seq_len)
        padded = F.pad(QEr, (1, 0))
        # padded.shape = (batch_size, num_heads, seq_len, 1 + seq_len)
        batch_size, num_heads, num_rows, num_cols = padded.shape
        reshaped = padded.reshape(batch_size, num_heads, num_cols, num_rows)
        # reshaped.size = (batch_size, num_heads, 1 + seq_len, seq_len)
        Srel = reshaped[:, :, 1:, :]
        # Srel.shape = (batch_size, num_heads, seq_len, seq_len)
        return Srel

    def forward(self, Q, K, V, attn_mask):

        if self.attn_type == "basic_attn" or not self.enc_attn:

            QEr = torch.einsum('bhqd, kd -> bhqk', Q, self.Er)
            Srel = self.skew(QEr)
            scores = (torch.einsum('bhqd, bhkd -> bhqk', Q, K)+Srel)/ np.sqrt(self.d_k)
            if attn_mask is not None:
                attn_mask = torch.as_tensor(attn_mask, dtype=torch.bool)
                attn_mask = attn_mask.to(self.device)
                scores.masked_fill_(attn_mask, -1e9)
            attn = self.softmax(scores)
            context = torch.einsum('bhqk,bhkd->bhqd', attn, V)

        elif "extra_info_attn" in self.attn_type:

            K_e = self.get_new_rep(K)
            V_e = self.get_new_rep(V)
            n = K_e.shape[2]
            linear_k = torch.sigmoid(self.linear_k)
            K_l = torch.einsum('bhnd,n->bhnd', K_e, linear_k) + \
                  torch.einsum('bhnd,n->bhnd', K[:, :, -n:, :].clone(), 1 - linear_k)
            V_l = torch.einsum('bhnd,n->bhnd', V_e, linear_k) + \
                  torch.einsum('bhnd,n->bhnd', V[:, :, -n:, :].clone(), 1 - linear_k)
            K[:, :, -n:, :] = K_l
            V[:, :, -n:, :] = V_l
            QEr = torch.einsum('bhqd, kd -> bhqk', Q, self.Er)
            Srel = torch.einsum('bhkq->bhqk', self.skew(QEr))
            scores = (torch.einsum('bhqd,bhkd-> bhqk', Q, K) + Srel) / np.sqrt(self.d_k)
            attn = self.softmax(scores)
            context = torch.einsum('bhqk,bhkd->bhqd', attn, V)

        return context, attn


class MultiHeadAttention(nn.Module):

    def __init__(self, d_model, d_k, d_v, n_heads, device, attn_type,
                 n_ext_info, kernel_s, kernel_b, enc_attn=False):

        super(MultiHeadAttention, self).__init__()

        self.WQ = nn.Linear(d_model, d_k * n_heads, bias=False)
        self.WK = nn.Linear(d_model, d_k * n_heads, bias=False)
        self.WV = nn.Linear(d_model, d_v * n_heads, bias=False)
        self.fc = nn.Linear(n_heads * d_v, d_model, bias=False)

        self.device = device

        self.d_model = d_model
        self.d_k = d_k
        self.d_v = d_v
        self.n_heads = n_heads
        self.attn_type = attn_type
        self.enc_attn = enc_attn
        self.n_ext_info = n_ext_info
        self.kernel_s = kernel_s
        self.kernel_b = kernel_b

    def forward(self, Q, K, V, attn_mask):

        batch_size = Q.shape[0]
        q_s = self.WQ(Q).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        k_s = self.WK(K).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        v_s = self.WV(V).view(batch_size, -1, self.n_heads, self.d_v).transpose(1, 2)
        if attn_mask is not None:
            attn_mask = attn_mask.unsqueeze(1).repeat(1, self.n_heads, 1, 1)
        context, attn = ScaledDotProductAttention(d_k=self.d_k,
                                                  device=self.device,
                                                  n_heads=self.n_heads,
                                                  l_k=k_s.shape[2],
                                                  b_size=batch_size,
                                                  n_ext_info=self.n_ext_info,
                                                  kernel_s=self.kernel_s,
                                                  kernel_b=self.kernel_b,
                                                  attn_type=self.attn_type,
                                                  enc_attn=self.enc_attn)(
            Q=q_s, K=k_s, V=v_s, attn_mask=attn_mask)
        context = context.transpose(1, 2).contiguous().view(batch_size, -1, self.n_heads * self.d_v)
        output = self.fc(context)
        return output, attn


class PoswiseFeedForwardNet(nn.Module):

    def __init__(self, d_model, d_ff):
        super(PoswiseFeedForwardNet, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)

    def forward(self, inputs):

        return self.w_2(F.relu(self.w_1(inputs)))


class EncoderLayer(nn.Module):

    def __init__(self, d_model, d_ff, d_k, d_v, n_heads, device, attn_type,
                 n_ext_info, kernel_s, kernel_b):
        super(EncoderLayer, self).__init__()
        self.enc_self_attn = MultiHeadAttention(
            d_model=d_model, d_k=d_k,
            d_v=d_v, n_heads=n_heads,
            device=device,
            attn_type=attn_type,
            n_ext_info=n_ext_info,
            kernel_s=kernel_s,
            kernel_b=kernel_b)
        self.pos_ffn = PoswiseFeedForwardNet(
            d_model=d_model, d_ff=d_ff)
        self.layer_norm = nn.LayerNorm(d_model, elementwise_affine=False)

    def forward(self, enc_inputs, enc_self_attn_mask=None):

        out, attn = self.enc_self_attn(
            Q=enc_inputs, K=enc_inputs,
            V=enc_inputs, attn_mask=enc_self_attn_mask)

        out = self.layer_norm(out + enc_inputs)
        out_2 = self.pos_ffn(out)
        out_2 = self.layer_norm(out_2 + out)
        return out_2, attn


class Encoder(nn.Module):

    def __init__(self, d_model, d_ff, d_k, d_v, n_heads,
                 n_layers, pad_index, device, attn_type,
                 n_ext_info, kernel_s, kernel_b):
        super(Encoder, self).__init__()
        self.device = device
        self.pad_index = pad_index
        self.pos_emb = PositionalEncoding(
            d_hid=d_model,
            device=device)
        self.n_layers = n_layers
        self.layers = []
        for _ in range(n_layers):
            encoder_layer = EncoderLayer(
                d_model=d_model, d_ff=d_ff,
                d_k=d_k, d_v=d_v, n_heads=n_heads,
                device=device, attn_type=attn_type,
                n_ext_info=n_ext_info,
                kernel_s=kernel_s, kernel_b=kernel_b)
            self.layers.append(encoder_layer)
        self.layers = nn.ModuleList(self.layers)

    def forward(self, enc_input):

        enc_outputs = enc_input

        enc_self_attn_mask = None

        enc_self_attns = []
        for layer in self.layers:
            enc_outputs, enc_self_attn = layer(enc_outputs, enc_self_attn_mask)
            enc_self_attns.append(enc_self_attn)

        enc_self_attns = torch.stack(enc_self_attns)
        enc_self_attns = enc_self_attns.permute([1, 0, 2, 3, 4])
        return enc_outputs, enc_self_attns


class DecoderLayer(nn.Module):

    def __init__(self, d_model, d_ff, d_k, d_v,
                 n_heads, device, attn_type,
                 n_ext_info, kernel_s, kernel_b):
        super(DecoderLayer, self).__init__()
        self.dec_self_attn = MultiHeadAttention(
            d_model=d_model, d_k=d_k,
            d_v=d_v, n_heads=n_heads, device=device,
            attn_type=attn_type, n_ext_info=n_ext_info,
            kernel_s=kernel_s, kernel_b=kernel_b)
        self.dec_enc_attn = MultiHeadAttention(
            d_model=d_model, d_k=d_k,
            d_v=d_v, n_heads=n_heads, device=device,
            attn_type=attn_type, n_ext_info=n_ext_info,
            kernel_s=kernel_s, kernel_b=kernel_b, enc_attn=True)
        self.pos_ffn = PoswiseFeedForwardNet(
            d_model=d_model, d_ff=d_ff)
        self.layer_norm = nn.LayerNorm(d_model, elementwise_affine=False)

    def forward(self, dec_inputs, enc_outputs,
                dec_self_attn_mask=None, dec_enc_attn_mask=None):

        out, dec_self_attn = self.dec_self_attn(Q=dec_inputs, K=dec_inputs, V=dec_inputs,
                                                              attn_mask=dec_self_attn_mask)
        out = self.layer_norm(dec_inputs + out)
        out2, dec_enc_attn = self.dec_enc_attn(Q=out, K=enc_outputs, V=enc_outputs,
                                               attn_mask=dec_enc_attn_mask)
        out2 = self.layer_norm(out + out2)
        out3 = self.pos_ffn(out2)
        out3 = self.layer_norm(out2 + out3)
        return out3, dec_self_attn, dec_enc_attn,


class Decoder(nn.Module):

    def __init__(self, d_model, d_ff, d_k, d_v,
                 n_heads, n_layers, pad_index,
                 device, attn_type, n_ext_info,
                 kernel_s, kernel_b):
        super(Decoder, self).__init__()
        self.pad_index = pad_index
        self.device = device
        self.pos_emb = PositionalEncoding(
            d_hid=d_model,
            device=device)
        self.layer_norm = nn.LayerNorm(d_model)
        self.layers = []
        for _ in range(n_layers):
            decoder_layer = DecoderLayer(
                d_model=d_model, d_ff=d_ff,
                d_k=d_k, d_v=d_v,
                n_heads=n_heads, device=device,
                attn_type=attn_type,
                n_ext_info=n_ext_info, kernel_s=kernel_s,
                kernel_b=kernel_b)
            self.layers.append(decoder_layer)
        self.layers = nn.ModuleList(self.layers)
        self.d_k = d_k

    def forward(self, dec_inputs, enc_outputs):

        dec_outputs = dec_inputs

        dec_self_attn_subsequent_mask = get_attn_subsequent_mask(dec_inputs)

        dec_self_attns, dec_enc_attns = [], []
        for layer in self.layers:
            dec_outputs, dec_self_attn, dec_enc_attn = layer(
                dec_inputs=dec_outputs,
                enc_outputs=enc_outputs,
                dec_self_attn_mask=dec_self_attn_subsequent_mask,
                dec_enc_attn_mask=None,
            )
            dec_self_attns.append(dec_self_attn)
            dec_enc_attns.append(dec_enc_attn)
        dec_self_attns = torch.stack(dec_self_attns)
        dec_enc_attns = torch.stack(dec_enc_attns)

        dec_self_attns = dec_self_attns.permute([1, 0, 2, 3, 4])
        dec_enc_attns = dec_enc_attns.permute([1, 0, 2, 3, 4])

        return dec_outputs, dec_self_attns, dec_enc_attns


class Attn(nn.Module):

    def __init__(self, src_input_size, tgt_input_size, d_model,
                 d_ff, d_k, d_v, n_heads, n_layers, src_pad_index,
                 tgt_pad_index, device, attn_type, n_ext_info,
                 kernel_s, kernel_b, seed):
        super(Attn, self).__init__()

        torch.manual_seed(seed)
        self.encoder = Encoder(
            d_model=d_model, d_ff=d_ff,
            d_k=d_k, d_v=d_v, n_heads=n_heads,
            n_layers=n_layers, pad_index=src_pad_index,
            device=device, attn_type=attn_type,
            n_ext_info=n_ext_info, kernel_s=kernel_s, kernel_b=kernel_b)
        self.decoder = Decoder(
            d_model=d_model, d_ff=d_ff,
            d_k=d_k, d_v=d_v, n_heads=n_heads,
            n_layers=n_layers, pad_index=tgt_pad_index,
            device=device, attn_type=attn_type,
            n_ext_info=n_ext_info, kernel_s=kernel_s, kernel_b=kernel_b)

        self.enc_embedding = nn.Linear(src_input_size, d_model)
        self.dec_embedding = nn.Linear(tgt_input_size, d_model)
        self.projection = nn.Linear(d_model, 1, bias=False)

    def forward(self, enc_inputs, dec_inputs):

        enc_inputs = self.enc_embedding(enc_inputs)
        dec_inputs = self.dec_embedding(dec_inputs)
        enc_outputs, enc_self_attns = self.encoder(enc_inputs)
        dec_outputs, dec_self_attns, dec_enc_attns = \
            self.decoder(dec_inputs, enc_outputs)
        dec_logits = self.projection(dec_outputs)
        return dec_logits

