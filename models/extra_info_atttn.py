import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import math
import random


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


class BasicAttn(nn.Module):

    def __init__(self, d_k, device):

        super(BasicAttn, self).__init__()
        self.device = device
        self.d_k = d_k
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, Q, K, V, attn_mask):

        scores = torch.einsum('bhqd, bhkd -> bhqk', Q, K) / np.sqrt(self.d_k)
        if attn_mask is not None:
            attn_mask = torch.as_tensor(attn_mask, dtype=torch.bool)
            attn_mask = attn_mask.to(self.device)
            scores.masked_fill_(attn_mask, -1e9)
        attn = self.softmax(scores)
        context = torch.einsum('bhqk,bhkd->bhqd', attn, V)

        return context, attn


class ACAT(nn.Module):

    def __init__(self, d_k, device, h):

        super(ACAT, self).__init__()
        self.device = device
        self.d_k = d_k

        self.filter_length_q = [3, 9, 14]
        self.filter_length_k = [3, 9, 14]
        self.conv_list_q = nn.ModuleList(
            [nn.Conv1d(in_channels=d_k*h, out_channels=d_k*h,
                       kernel_size=f,
                       padding=int(f/2),
                       bias=False,
                       device=device) for f in self.filter_length_q])
        self.conv_list_k = nn.ModuleList(
            [nn.Conv1d(in_channels=d_k*h, out_channels=d_k*h,
                       kernel_size=f,
                       padding=int(f/2),
                       bias=False,
                       device=device) for f in self.filter_length_k])
        self.norm = nn.BatchNorm1d(d_k*h, device=device)
        self.activation = nn.ELU()
        self.max_pooling = nn.MaxPool2d(kernel_size=(1, 3))

        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='leaky_relu')

    def forward(self, Q, K, V, attn_mask):

        b, h, l, d_k = Q.shape
        l_k = K.shape[2]

        Q_l = [self.activation(self.norm(
               self.conv_list_q[i](Q.reshape(b, h*d_k, -1))))
               [:, :, :l].reshape(b, l, -1)
               for i in range(len(self.filter_length_q))]
        K_l = [self.activation(self.norm(
               self.conv_list_k[i](K.reshape(b, h*d_k, -1))))
               [:, :, :l_k].reshape(b, l_k, -1)
               for i in range(len(self.filter_length_k))]

        Q_p = torch.cat(Q_l, dim=0).reshape(b, d_k*h, l, -1)
        K_p = torch.cat(K_l, dim=0).reshape(b, d_k*h, l_k, -1)
        Q = self.max_pooling(Q_p).reshape(b, h, -1, d_k)
        K = self.max_pooling(K_p).reshape(b, h, -1, d_k)
        log_k = int(l_k / max(self.filter_length_k))
        K, index = torch.topk(K, dim=2, k=log_k)
        index = torch.max(index, -1)[0]
        index = index.unsqueeze(-2).repeat(1, 1, l, 1)

        scores = torch.einsum('bhqd,bhkd->bhqk', Q, K) / np.sqrt(self.d_k)

        if attn_mask is not None:
            attn_mask = attn_mask[:, :, :, 0:log_k]
            attn_mask = torch.as_tensor(attn_mask, dtype=torch.bool)
            attn_mask = attn_mask.to(self.device)
            scores.masked_fill_(attn_mask, -1e9)

        attn = torch.softmax(scores, -1)

        attn_f = torch.zeros(b, h, l, l_k, device=self.device)
        attn_f[torch.arange(b)[:, None, None, None],
               torch.arange(h)[None, :, None, None],
               torch.arange(l)[None, None, :, None],
               index] = attn
        context = torch.einsum('bhqk,bhkd->bhqd', attn_f, V)
        return context, attn


class MultiHeadAttention(nn.Module):

    def __init__(self, d_model, d_k, d_v, n_heads, attn_type, device):

        super(MultiHeadAttention, self).__init__()

        self.WQ = nn.Linear(d_model, d_k * n_heads, bias=False)
        self.WK = nn.Linear(d_model, d_k * n_heads, bias=False)
        self.WV = nn.Linear(d_model, d_k * n_heads, bias=False)
        self.fc = nn.Linear(n_heads * d_v, d_model, bias=False)

        self.device = device

        self.d_model = d_model
        self.d_k = d_k
        self.d_v = d_v
        self.n_heads = n_heads
        self.attn_type = attn_type

    def forward(self, Q, K, V, attn_mask):

        batch_size = Q.shape[0]
        q_s = self.WQ(Q).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        k_s = self.WK(K).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        v_s = self.WV(V).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        if attn_mask is not None:
            attn_mask = attn_mask.unsqueeze(1).repeat(1, self.n_heads, 1, 1)
        if self.attn_type == "BasicAttn":
            context, attn = BasicAttn(d_k=self.d_k, device=self.device)(
            Q=q_s, K=k_s, V=v_s, attn_mask=attn_mask)
        elif self.attn_type == "ACAT":
            context, attn = ACAT(d_k=self.d_k, device=self.device,
                                 h=self.n_heads)(
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

    def __init__(self, d_model, d_ff, d_k, d_v, n_heads, device, dr, attn_type):
        super(EncoderLayer, self).__init__()
        self.enc_self_attn = MultiHeadAttention(
            d_model=d_model, d_k=d_k,
            d_v=d_v, n_heads=n_heads,
            device=device, attn_type=attn_type)
        self.pos_ffn = PoswiseFeedForwardNet(
            d_model=d_model, d_ff=d_ff)
        self.layer_norm = nn.LayerNorm(d_model, elementwise_affine=False)
        self.dropout = nn.Dropout(p=dr)

    def forward(self, enc_inputs, enc_self_attn_mask=None):

        out, attn = self.enc_self_attn(
            Q=enc_inputs, K=enc_inputs,
            V=enc_inputs, attn_mask=enc_self_attn_mask)

        out = self.layer_norm(out + enc_inputs)
        out_2 = self.pos_ffn(out)
        out_2 = self.layer_norm(out_2 + out)
        out_2 = self.dropout(out_2)

        return out_2, attn


class Encoder(nn.Module):

    def __init__(self, d_model, d_ff, d_k, d_v, n_heads,
                 n_layers, pad_index, device,  dr, attn_type):
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
                device=device, dr=dr, attn_type=attn_type)
            self.layers.append(encoder_layer)
        self.layers = nn.ModuleList(self.layers)

    def forward(self, enc_input):

        enc_outputs = self.pos_emb(enc_input)

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
                 n_heads, attn_type, device, dr):
        super(DecoderLayer, self).__init__()
        self.dec_self_attn = MultiHeadAttention(
            d_model=d_model, d_k=d_k,
            d_v=d_v, n_heads=n_heads, device=device, attn_type=attn_type)
        self.dec_enc_attn = MultiHeadAttention(
            d_model=d_model, d_k=d_k,
            d_v=d_v, n_heads=n_heads, device=device, attn_type=attn_type)
        self.pos_ffn = PoswiseFeedForwardNet(
            d_model=d_model, d_ff=d_ff)
        self.layer_norm = nn.LayerNorm(d_model, elementwise_affine=False)
        self.dropout = nn.Dropout(p=dr)

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
        out3 = self.dropout(out3)
        return out3, dec_self_attn, dec_enc_attn,


class Decoder(nn.Module):

    def __init__(self, d_model, d_ff, d_k, d_v,
                 n_heads, n_layers, pad_index,
                 device, attn_type, dr):
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
                attn_type=attn_type, dr=dr)
            self.layers.append(decoder_layer)
        self.layers = nn.ModuleList(self.layers)
        self.d_k = d_k

    def forward(self, dec_inputs, enc_outputs):

        dec_outputs = self.pos_emb(dec_inputs)

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
                 tgt_pad_index, device, attn_type, dr, seed):
        super(Attn, self).__init__()

        np.random.seed(seed)
        random.seed(seed)
        torch.manual_seed(seed)
        self.encoder = Encoder(
            d_model=d_model, d_ff=d_ff,
            d_k=d_k, d_v=d_v, n_heads=n_heads,
            n_layers=n_layers, pad_index=src_pad_index,
            device=device, attn_type=attn_type, dr=dr)
        self.decoder = Decoder(
            d_model=d_model, d_ff=d_ff,
            d_k=d_k, d_v=d_v, n_heads=n_heads,
            n_layers=n_layers, pad_index=tgt_pad_index,
            device=device, attn_type=attn_type, dr=dr)

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

