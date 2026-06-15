import torch 
import torch.nn as nn
import math

##Helper
def create_sin_cos_matrix(num_embeddings, d_model):
    weight = torch.zeros(num_embeddings, d_model)
    weight.requires_grad = False
    
    position = torch.arange(0, num_embeddings, dtype=torch.float).unsqueeze(1)
    div_term = torch.exp(torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model))
    
    weight[:, 0::2] = torch.sin(position * div_term)
    weight[:, 1::2] = torch.cos(position * div_term)
    return weight

class PositionalEmbedding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEmbedding, self).__init__()
        pe = create_sin_cos_matrix(max_len, d_model).unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return self.pe[:, :x.size(1)]
    
class ValueEmbedding(nn.Module):
    def __init__(self, c_in, d_model):
        super(ValueEmbedding, self).__init__()
        padding = 1 if torch.__version__ >= '1.5.0' else 2
        self.tokenConvolution = nn.Conv1d(in_channels=c_in, out_channels=d_model, kernel_size=3, padding=padding, padding_mode='circular', bias=False)
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='leaky_relu')
                
    def forward(self, a):
        a = self.tokenConvolution(a.permute(0, 2, 1)).permute(0, 2, 1)
        return a
    
class TemporalEmbedding(nn.Module):
    def __init__(self, d_model, embed_type='fixed'):
        super(TemporalEmbedding, self).__init__()
        
        hour_size, day_size, weekday_size = 24, 32, 7

        if embed_type == 'fixed':
            self.hour_embed = nn.Embedding(hour_size, d_model)
            self.hour_embed.weight = nn.Parameter(create_sin_cos_matrix(hour_size, d_model), requires_grad=False)
            
            self.day_embed = nn.Embedding(day_size, d_model)
            self.day_embed.weight = nn.Parameter(create_sin_cos_matrix(day_size, d_model), requires_grad=False)
            
            self.weekday_embed = nn.Embedding(weekday_size, d_model)
            self.weekday_embed.weight = nn.Parameter(create_sin_cos_matrix(weekday_size, d_model), requires_grad=False)
        else:
            self.hour_embed = nn.Embedding(hour_size, d_model)
            self.day_embed = nn.Embedding(day_size, d_model)
            self.weekday_embed = nn.Embedding(weekday_size, d_model)
    def forward(self, x_mark):
        x_mark = x_mark.long()
        return self.hour_embed(x_mark[:, :, 0]) + self.day_embed(x_mark[:, :, 1]) + self.weekday_embed(x_mark[:, :, 2])

class DataEmbedding(nn.Module):
    def __init__(self, c_in, d_model, embed_type='fixed', max_len=5000, dropout=0.1):
        super(DataEmbedding, self).__init__()

        self.value_embedding = ValueEmbedding(c_in=c_in, d_model=d_model)
        self.position_embedding = PositionalEmbedding(d_model=d_model, max_len=max_len)
        self.temporal_embedding = TemporalEmbedding(d_model=d_model, embed_type=embed_type)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, a, a_mark):
        return self.dropout(self.value_embedding(a) + self.position_embedding(a) + self.temporal_embedding(a_mark))

if __name__ == "__main__":
    batch_size = 2
    seq_len = 96
    c_in = 7      
    d_model = 512 

    x = torch.randn(batch_size, seq_len, c_in)
    
    x_mark = torch.zeros(batch_size, seq_len, 3)
    x_mark[:, :, 0] = torch.randint(0, 24, (batch_size, seq_len))
    x_mark[:, :, 1] = torch.randint(1, 32, (batch_size, seq_len))
    x_mark[:, :, 2] = torch.randint(0, 7, (batch_size, seq_len))

    embedding_layer = DataEmbedding(c_in=c_in, d_model=d_model, embed_type='fixed')
    
    output = embedding_layer(x, x_mark)
    
    print("Size of input x:", x.shape)
    print("Size of input x_mark:", x_mark.shape)
    print("Size of output after DataEmbedding:", output.shape)
