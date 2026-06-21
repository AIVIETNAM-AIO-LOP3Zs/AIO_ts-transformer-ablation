import torch 
import torch.nn as nn
import math

def create_sin_cos_matrix(num_embeddings, d_model):
    # Initialize a matrix of zeros with shape (num_embeddings, d_model)
    weight = torch.zeros(num_embeddings, d_model)
    weight.requires_grad = False  # Freeze gradients since this is a static formula
    
    # Create a column vector for positions: [0, 1, 2, ..., num_embeddings-1] -> Shape: (num_embeddings, 1)
    position = torch.arange(0, num_embeddings, dtype=torch.float).unsqueeze(1)
    
    # Compute the scaling/denominator term for frequencies: 10000^(2i/d_model)
    # Applied to every 2nd index, matching the standard Transformer design
    div_term = torch.exp(torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model))
    
    # Fill even indices (0, 2, 4...) with sine waves
    weight[:, 0::2] = torch.sin(position * div_term)
    # Fill odd indices (1, 3, 5...) with cosine waves.
    # Slice div_term to the number of odd columns so an *odd* d_model (where
    # there are fewer odd than even indices) does not raise a shape mismatch.
    weight[:, 1::2] = torch.cos(position * div_term[: weight[:, 1::2].size(1)])
    return weight

class PositionalEmbedding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEmbedding, self).__init__()
        # Generate the positional matrix and add a batch dimension: Shape: (1, max_len, d_model)
        pe = create_sin_cos_matrix(max_len, d_model).unsqueeze(0)
        # register_buffer saves this tensor in the module state dict but ensures it won't be trained
        self.register_buffer('pe', pe)

    def forward(self, x):
        # Slice the pre-computed matrix to match the current input sequence length: x.size(1)
        # Output shape: (1, seq_len, d_model)
        return self.pe[:, :x.size(1)]
    
class ValueEmbedding(nn.Module):
    def __init__(self, c_in, d_model):
        super(ValueEmbedding, self).__init__()
        # kernel_size=3 with circular padding=1 keeps the sequence length unchanged
        # (out_len = L + 2*padding - kernel_size + 1 = L). Hardcoded because the old
        # string comparison `torch.__version__ >= '1.5.0'` was lexicographic and wrong
        # for versions like '1.10.0' (compares as < '1.5.0').
        padding = 1

        # Use a 1D Convolution over the time steps to project raw features (c_in) to embedding space (d_model)
        self.tokenConvolution = nn.Conv1d(in_channels=c_in, out_channels=d_model, 
                                          kernel_size=3, padding=padding, 
                                          padding_mode='circular', bias=False)
        
        # Initialize weights using Kaiming Normal method (ideal for ReLU/LeakyReLU activations)
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='leaky_relu')
                
    def forward(self, a):
        # nn.Conv1d expects input shapes as (batch, channels, length). 
        # 1. permute(0, 2, 1) changes (B, L, C) -> (B, C, L)
        # 2. Apply convolution -> (B, d_model, L)
        # 3. permute(0, 2, 1) restores it back -> (B, L, d_model)
        a = self.tokenConvolution(a.permute(0, 2, 1)).permute(0, 2, 1)
        return a
    
class TemporalEmbedding(nn.Module):
    def __init__(self, d_model, embed_type='fixed'):
        super(TemporalEmbedding, self).__init__()
        
        # Define ranges for time-series metadata categories
        hour_size, day_size, weekday_size = 24, 32, 7

        # 'fixed' uses deterministic sin/cos matrices. 'learned' uses trainable parameters.
        if embed_type == 'fixed':
            self.hour_embed = nn.Embedding(hour_size, d_model)
            self.hour_embed.weight = nn.Parameter(create_sin_cos_matrix(hour_size, d_model), requires_grad=False)
            
            self.day_embed = nn.Embedding(day_size, d_model)
            self.day_embed.weight = nn.Parameter(create_sin_cos_matrix(day_size, d_model), requires_grad=False)
            
            self.weekday_embed = nn.Embedding(weekday_size, d_model)
            self.weekday_embed.weight = nn.Parameter(create_sin_cos_matrix(weekday_size, d_model), requires_grad=False)
        else:
            # Default trainable embeddings
            self.hour_embed = nn.Embedding(hour_size, d_model)
            self.day_embed = nn.Embedding(day_size, d_model)
            self.weekday_embed = nn.Embedding(weekday_size, d_model)
            
    def forward(self, x_mark):
        # Convert metadata inputs to integers (long) to look up indices in embedding tables
        x_mark = x_mark.long()
        
        # Extract and sum up temporal elements: Hour + Day of Month + Day of Week
        # Each lookup yields a (Batch, Seq_len, d_model) tensor
        return self.hour_embed(x_mark[:, :, 0]) + self.day_embed(x_mark[:, :, 1]) + self.weekday_embed(x_mark[:, :, 2])

class DataEmbedding(nn.Module):
    """
    Combines Value (Feature) Embedding, Global Positional Embedding, 
    and Temporal (Calendar) Embedding by adding them together. Common in Informer/Transformer-based time-series models.
    """
    def __init__(self, c_in, d_model, embed_type='fixed', max_len=5000, dropout=0.1):
        super(DataEmbedding, self).__init__()

        self.value_embedding = ValueEmbedding(c_in=c_in, d_model=d_model)
        self.position_embedding = PositionalEmbedding(d_model=d_model, max_len=max_len)
        self.temporal_embedding = TemporalEmbedding(d_model=d_model, embed_type=embed_type)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, a, a_mark):
        # Sum the three distinct embeddings together (element-wise addition) and apply dropout
        # Due to PyTorch broadcasting, the position embedding (1, L, D) automatically stretches to match the Batch size.
        return self.dropout(self.value_embedding(a))

if __name__ == "__main__":
    # Mock parameters representing a typical Informer/Autoformer pipeline execution
    batch_size = 2
    seq_len = 96
    c_in = 7         # Number of input variables/features (e.g., Weather metrics like temp, humidity, pressure...)
    d_model = 512    # Target Latent/Embedding space dimension

    # Mock raw sequential input data: (Batch, Sequence Length, Features)
    x = torch.randn(batch_size, seq_len, c_in)
    
    # Mock calendar metadata marks: (Batch, Sequence Length, 3) 
    # Index 0: Hour of day, Index 1: Day of month, Index 2: Day of week
    x_mark = torch.zeros(batch_size, seq_len, 3)
    x_mark[:, :, 0] = torch.randint(0, 24, (batch_size, seq_len))   # Hours (0-23)
    x_mark[:, :, 1] = torch.randint(1, 32, (batch_size, seq_len))   # Days (1-31)
    x_mark[:, :, 2] = torch.randint(0, 7, (batch_size, seq_len))    # Weekdays (0-6)

    # Initialize the composite Data Embedding Layer
    embedding_layer = DataEmbedding(c_in=c_in, d_model=d_model, embed_type='fixed')
    
    # Run data through forward pass
    output = embedding_layer(x, x_mark)
    
    
    print("Size of input x:", x.shape)                       
    print("Size of input x_mark:", x_mark.shape)             
    print("Size of output after DataEmbedding:", output.shape)