import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
from .dilated_conv import DilatedConvEncoder
import tensorflow as tf

def generate_continuous_mask(B, T, n=5, l=0.1):
    res = torch.full((B, T), True, dtype=torch.bool)
    if isinstance(n, float):
        n = int(n * T)
    n = max(min(n, T // 2), 1)
    
    if isinstance(l, float):
        l = int(l * T)
    l = max(l, 1)
    
    for i in range(B):
        for _ in range(n):
            t = np.random.randint(T-l+1)
            res[i, t:t+l] = False
    return res

def generate_binomial_mask(B, T, p=0.5):
    return torch.from_numpy(np.random.binomial(1, p, size=(B, T))).to(torch.bool)

global window_size
window_size = 1

#ここでスライディングウィンドウありかどうか決定する！
slide=True
#slide=False

class TSEncoder(nn.Module):
    def __init__(self, input_dims, output_dims, length_dim=10,hidden_dims=64, depth=10, mask_mode='binomial',input_total=1):
        super().__init__()
        self.input_dims = input_dims
        self.output_dims = output_dims
        self.hidden_dims = hidden_dims
        self.mask_mode = mask_mode
        print(input_total)
        print(input_dims*input_total)

        self.input_regime=nn.Linear(window_size,1)
        self.input_regime_3=nn.Linear(3,1)
        self.input_regime_5=nn.Linear(5,1)
        self.input_regime_7=nn.Linear(7,1)
        # self.input_regime=nn.Linear(window_size,10)
        # self.input_regime2=nn.Linear(10,1)
        #self.input_fc = nn.Linear(input_dims*input_total, hidden_dims)

        self.input_fc = nn.Linear(input_dims, hidden_dims)

        self.feature_extractor = DilatedConvEncoder(
            hidden_dims,
            [hidden_dims] * depth + [output_dims],
            kernel_size=3
        )
        #self.attention = nn.MultiheadAttention(embed_dim=output_dims, num_heads=4)
        self.repr_dropout = nn.Dropout(p=0.1)
        
    def kaiso(self, num,x):
        nan_mask = ~x.isnan().any(axis=-1)
        x[~nan_mask] = 0
        
        #ここがスライディングウィンドウ
        # 分割したデータを格納するリスト
        split_data = []
        if num==1:
            window_size=3
        elif num==2:   
            window_size=5
        elif num==3:
            window_size=7
        else:
            window_size=1
        if slide:
        # スライディングウィンドウで2次元目を分割
            for i in range(x.shape[1] - window_size + 1):
                split = x[:, i:i+window_size, :]
                x_window=split.transpose(1, 2)
                # print("test")
                # print(x_window.shape)
                if num==1:
                    y=self.input_regime_3(x_window)
                elif num==2:
                    y=self.input_regime_5(x_window)
                elif num==3:    
                    y=self.input_regime_7(x_window)
                else:
                    y=self.input_regime(x_window)
                # y=F.relu(self.input_regime(x_window))
                # y=self.input_regime2(y)
                y=y.transpose(1, 2)
                split_data.append(y)

            # 分割されたデータを結合する
            x = torch.cat(split_data, axis=1)
        # print("slide_size")
        # print(x.shape)
        #linearは最後の次元の層が変換される
        x = self.input_fc(x)
        #print(x.shape)
        #x = self.input_fc(x.transpose(1, 2))  # B x T x Ch
        # print("c")
        # print(x.shape)
        # generate & apply mask
        mask=None
        if mask is None:
            if self.training:
                mask = self.mask_mode
            else:
                mask = 'all_true'
        
        if mask == 'binomial':
            mask = generate_binomial_mask(x.size(0), x.size(1)).to(x.device)
        elif mask == 'continuous':
            mask = generate_continuous_mask(x.size(0), x.size(1)).to(x.device)
        elif mask == 'all_true':
            mask = x.new_full((x.size(0), x.size(1)), True, dtype=torch.bool)
        elif mask == 'all_false':
            mask = x.new_full((x.size(0), x.size(1)), False, dtype=torch.bool)
        elif mask == 'mask_last':
            mask = x.new_full((x.size(0), x.size(1)), True, dtype=torch.bool)
            mask[:, -1] = False

        nan_mask = ~x.isnan().any(axis=-1)
        mask &= nan_mask
        x[~mask] = 0
        
        # conv encoder
        x = x.transpose(1, 2)  # B x Ch x T
        x = self.repr_dropout(self.feature_extractor(x))  # B x Co x T
        #x, _ = self.attention(x, x, x)
        x = x.transpose(1, 2)  # B x T x Co
        # print(x)

        
        batch_size = x.shape[2]
        train_dataset_size = x.shape[1]
        k_cluster = 6
        lambda_kmeans =1e-3
        x_h=x.transpose(1, 2)
        x_numpy=x_h[0].cpu().detach().numpy().copy()
        # print(x.shape)
        #print(x_numpy.shape)

        # print(batch_size)
        #loss_km

        # HTH = tf.matmul(x_numpy,tf.transpose(x_numpy))
        # F_copy = tf.compat.v1.get_variable('F', shape=[batch_size, k_cluster],
        #                 initializer=tf.compat.v1.orthogonal_initializer(gain=1.0, seed=None, dtype=tf.float32),
        #                 dtype=tf.float32,
        #                 trainable=False
        #                 )
        
        # PyTorchのコード
        x_torch = torch.from_numpy(x_numpy)
        HTH = torch.matmul(x_torch, x_torch.t())

        F_copy = torch.empty(batch_size, k_cluster, dtype=torch.float32)
        torch.nn.init.orthogonal_(F_copy, gain=1.0)
        F_copy.requires_grad_(False)

        # PyTorchのコード
        FTHTHF = torch.matmul(torch.matmul(F_copy.t(), HTH), F_copy)
        loss_km = torch.trace(HTH) - torch.trace(FTHTHF)
        global_step = torch.tensor(0, requires_grad=False)
        lbdkm = 0.05
        loss_k = loss_km * lbdkm
        
        return x,loss_k

    def forward(self, x, mask=None):  # x: B x T x input_dims
        # print("a")
        # print(x.shape)
        x1,loss_k_1 = self.kaiso(5,x)
        return x1,loss_k_1
        # x1,loss_k_1 = self.kaiso(1,x)
        # x2,loss_k_2 = self.kaiso(2,x)
        # x3,loss_k_3 = self.kaiso(3,x)
        # return x1,x2,x3,loss_k_1,loss_k_2,loss_k_3
        
        