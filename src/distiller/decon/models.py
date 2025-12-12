import torch
import random, pdb
import warnings
import numpy as np
from pandas import DataFrame
from tqdm import tqdm
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
warnings.filterwarnings("ignore")
from .utils import device

# class SimpleDataset(Dataset):
#     def __init__(self, X, Y):
#         self.X = X
#         self.Y = Y

#     def __len__(self):
#         return len(self.X)

#     def __getitem__(self, index):
#         if type(self.X)==np.ndarray:
#             x = torch.from_numpy(self.X[index]).float().to(device)
#         elif type(self.X)==DataFrame:
#             x = torch.from_numpy(self.X.values[index]).float().to(device)
        
#         if type(self.Y)==np.ndarray:
#             y = torch.from_numpy(self.Y[index]).float().to(device)
#         elif type(self.Y)==DataFrame:
#             y = torch.from_numpy(self.Y.values[index]).float().to(device)
#         return x, y


class SimpleDataset(Dataset):
    def __init__(self, X, Y=None):
        self.X = X
        self.Y = Y  

    def __len__(self):
        return len(self.X)

    def __getitem__(self, index):
        if type(self.X) == np.ndarray:
            x = torch.from_numpy(self.X[index]).float().to(device)
        elif type(self.X) == DataFrame:
            x = torch.from_numpy(self.X.values[index]).float().to(device)

        if self.Y is not None:  
            if type(self.Y) == np.ndarray:
                y = torch.from_numpy(self.Y[index]).float().to(device)
            elif type(self.Y) == DataFrame:
                y = torch.from_numpy(self.Y.values[index]).float().to(device)
            return x, y
        else:
            return x  


### TAPE's AutoEncoder
class AutoEncoder(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.name = 'ae'
        self.state = 'train' # or 'test'
        self.inputdim = input_dim
        self.outputdim = output_dim
        self.encoder = nn.Sequential(nn.Dropout(), nn.Linear(self.inputdim, 512), nn.CELU(),
                                     nn.Dropout(), nn.Linear(512, 256), nn.CELU(),
                                     nn.Dropout(), nn.Linear(256, 128), nn.CELU(),
                                     nn.Dropout(), nn.Linear(128, 64), nn.CELU(),
                                     nn.Linear(64, output_dim),
                                     )

        self.decoder = nn.Sequential(nn.Linear(self.outputdim, 64, bias=False),
                                     nn.Linear(64, 128, bias=False),
                                     nn.Linear(128, 256, bias=False),
                                     nn.Linear(256, 512, bias=False),
                                     nn.Linear(512, self.inputdim, bias=False))

    def encode(self, x):
        return self.encoder(x)

    def decode(self, z):
        return self.decoder(z)

    def refraction(self,x):
        x_sum = torch.sum(x, dim=1, keepdim=True)
        return x/x_sum
    
    def sigmatrix(self):
        w0 = (self.decoder[0].weight.T)
        w1 = (self.decoder[1].weight.T)
        w2 = (self.decoder[2].weight.T)
        w3 = (self.decoder[3].weight.T)
        w4 = (self.decoder[4].weight.T)
        w01 = (torch.mm(w0, w1))
        w02 = (torch.mm(w01, w2))
        w03 = (torch.mm(w02, w3))
        w04 = (torch.mm(w03, w4))
        return nn.functional.relu(w04)

    def forward(self, x):
        sigmatrix = self.sigmatrix()
        z = self.encode(x)
        if self.state == 'train':
            pass
        elif self.state == 'test':
            z = nn.functional.relu(z)
            z = self.refraction(z)

        x_recon = torch.mm(z, sigmatrix)
        return x_recon, z, sigmatrix
    
    def rebuild(self, z):
        sigmatrix = self.sigmatrix()
        # print(sigmatrix)
        x_recon = torch.mm(z, sigmatrix)
        return x_recon

### Customized layer to normalize the output of the AutoEncoder
class CustomNormalization(nn.Module):
    def __init__(self):
        super(CustomNormalization, self).__init__()

    def forward(self, x):
        return x / (x.sum(dim=1, keepdim=True) + 1e-8)
    

class CustomMaxAbsNormalization(nn.Module):
    def __init__(self):
        super(CustomMaxAbsNormalization, self).__init__()

    def forward(self, x):
        max_abs_values = x.abs().max(dim=1, keepdim=True).values
        return x / max_abs_values


### A modified version of AutoEncoder, which adds a ReLU layer and a normalization layer to the output of the encoder for outputing a probability
class AutoEncoderPlus(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.inputdim = input_dim
        self.outputdim = output_dim
        self.encoder = nn.Sequential(nn.Dropout(p=0.2), nn.Linear(self.inputdim, 512), nn.CELU(),
                                nn.Dropout(p=0.2), nn.Linear(512, 256), nn.CELU(),
                                nn.Dropout(p=0.2), nn.Linear(256, 128), nn.CELU(),
                                nn.Dropout(p=0.2), nn.Linear(128, 64), nn.CELU(),
                                nn.Linear(64, output_dim), 
                                nn.ReLU(), CustomNormalization()
                            )
        # self.encoder = nn.Sequential(nn.Dropout(p=0.2), nn.Linear(self.inputdim, 512), nn.CELU(),
        #                              nn.Dropout(p=0.2), nn.Linear(512, 256), nn.CELU(),
        #                              nn.Dropout(p=0.2), nn.Linear(256, 128), nn.CELU(),
        #                              nn.Dropout(p=0.2), nn.Linear(128, 64), nn.CELU(),
        #                              nn.Linear(64, output_dim), 
        #                              CustomMaxAbsNormalization(),
        #                              nn.Softmax()
        #                             )
        
        self.decoder = nn.Sequential(#F.log_softmax(),
                                     nn.Linear(self.outputdim, 64, bias=False), nn.CELU(),
                                     nn.Linear(64, 128, bias=False), nn.CELU(),
                                     nn.Linear(128, 256, bias=False), nn.CELU(),
                                     nn.Linear(256, 512, bias=False), nn.CELU(),
                                     nn.Linear(512, self.inputdim, bias=False))
        
        self.initialize_weights()

    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def encode(self, x):
        return self.encoder(x)

    def decode(self, z):
        return self.decoder(z)

    def refraction(self,x):
        x_sum = torch.sum(x, dim=1, keepdim=True)
        return x/x_sum
    
    ### Return the signature matrix
    def sigmatrix(self):
        # Extract all Linear layers from decoder to avoid indexing activation layers
        linear_layers = [layer for layer in self.decoder if isinstance(layer, nn.Linear)]
        
        # Get weight matrices from the 5 Linear layers
        w0 = (linear_layers[0].weight.T)
        w1 = (linear_layers[1].weight.T)
        w2 = (linear_layers[2].weight.T)
        w3 = (linear_layers[3].weight.T)
        w4 = (linear_layers[4].weight.T)
        
        # Compute the full signature matrix by multiplying all weight matrices
        w01 = (torch.mm(w0, w1))
        w02 = (torch.mm(w01, w2))
        w03 = (torch.mm(w02, w3))
        w04 = (torch.mm(w03, w4))
        return nn.functional.relu(w04)
    
    ### Return the signature matrix without first Layer (cell_types -> 64) to help calculate the residual part
    def raw_sigmatrix2(self):
        # Extract all Linear layers from decoder to avoid indexing activation layers
        linear_layers = [layer for layer in self.decoder if isinstance(layer, nn.Linear)]
        
        # Skip the first Linear layer and use the remaining 4 layers
        w1 = (linear_layers[1].weight.T)
        w2 = (linear_layers[2].weight.T)
        w3 = (linear_layers[3].weight.T)
        w4 = (linear_layers[4].weight.T)
        
        # Compute the partial signature matrix
        w12 = (torch.mm(w1, w2))
        w13 = (torch.mm(w12, w3))
        w14 = (torch.mm(w13, w4))
        return w14

    def rebuild(self, z):
        sigmatrix = self.sigmatrix()
        # print(sigmatrix)
        x_recon = torch.mm(z, sigmatrix)
        return x_recon
    
    
    def update_decoder_weights(self, ae_model, infer_sig):
        """
        Update decoder weights based on another AutoEncoder model and the infer_sig matrix.
        """
        # Extract all Linear layers from both models
        self_linear_layers = [layer for layer in self.decoder if isinstance(layer, nn.Linear)]
        ae_linear_layers = [layer for layer in ae_model.decoder if isinstance(layer, nn.Linear)]
        
        # Update weights for all Linear layers
        for idx, (self_layer, ae_layer) in enumerate(zip(self_linear_layers, ae_linear_layers)):
            if idx > 0:
                # Copy weights from the corresponding layer in ae_model
                self_layer.weight.data = ae_layer.weight.data.clone()
            elif idx == 0:
                # For the first layer, copy existing weights and add new signature
                self_layer.weight.data[:, :-1] = ae_layer.weight.data.clone()
                self_layer.weight.data[:, -1] = torch.from_numpy(infer_sig)

    def forward(self, x):
        sigmatrix = self.sigmatrix()
        z = self.encode(x)

        x_recon = torch.mm(z, sigmatrix)
        return x_recon, z, sigmatrix


### Decoder part of the AutoEncoder
class DecoderOnly(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.inputdim = input_dim
        self.outputdim = output_dim

        self.decoder = nn.Sequential(nn.Linear(self.outputdim, 64, bias=False),
                                     nn.Linear(64, 128, bias=False),
                                     nn.Linear(128, 256, bias=False),
                                     nn.Linear(256, 512, bias=False),
                                     nn.Linear(512, self.inputdim, bias=False))
        self.init_weights()

    def init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.kaiming_uniform(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

    def decode(self, z):
        return self.decoder(z)

    def sigmatrix(self):
        w0 = (self.decoder[0].weight.T)
        w1 = (self.decoder[1].weight.T)
        w2 = (self.decoder[2].weight.T)
        w3 = (self.decoder[3].weight.T)
        w4 = (self.decoder[4].weight.T)
        w01 = (torch.mm(w0, w1))
        w02 = (torch.mm(w01, w2))
        w03 = (torch.mm(w02, w3))
        w04 = (torch.mm(w03, w4))
        return nn.functional.relu(w04)
    
    def raw_sigmatrix2(self):
        w1 = (self.decoder[1].weight.T)
        w2 = (self.decoder[2].weight.T)
        w3 = (self.decoder[3].weight.T)
        w4 = (self.decoder[4].weight.T)
        
        w12 = (torch.mm(w1, w2))
        w13 = (torch.mm(w12, w3))
        w14 = (torch.mm(w13, w4))
        return w14

    def forward(self, z):
        return self.decoder(z)
    

### Encoder part of the AutoEncoder
class EncoderOnly(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.inputdim = input_dim
        self.outputdim = output_dim

        self.encoder = nn.Sequential(nn.Dropout(p=0.3),
                                     nn.Linear(self.inputdim, 2048),
                                     nn.CELU(),
                                     
                                    nn.Dropout(p=0.3),
                                    nn.Linear(2048, 512),
                                    nn.CELU(),
                                     
                                    nn.Dropout(p=0.3),
                                    nn.Linear(512, 128),
                                    nn.CELU(),
                                     
                                    nn.Dropout(p=0.3),
                                    nn.Linear(128, 64),
                                    nn.CELU(),
                                     
                                    nn.Linear(64, output_dim),
                                    nn.ReLU(),
                                    CustomNormalization()
                                    )
        
        self.init_weights()

    def init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.kaiming_uniform(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

    def forward(self, x):
        z = self.encoder(x)
        return z

class MLP_cls(nn.Module):
    def __init__(self, input_size, hidden_sizes):
        super().__init__()
        layers = []
        layers.append(nn.Linear(input_size, hidden_sizes[0]).to(device))
        layers.append(nn.LeakyReLU(0.2))

        for i in range(len(hidden_sizes)-1):
            # layers.append(nn.Dropout(p=0.2))
            layers.append(nn.Linear(hidden_sizes[i], hidden_sizes[i+1]).to(device))
            layers.append(nn.LeakyReLU(0.2))

        layers.append(nn.Linear(hidden_sizes[-1],1).to(device))
        layers.append(nn.Sigmoid())

        self.layers = nn.Sequential(*layers)

    def initialize_weights(self,device=None):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight.to(device))
                if m.bias is not None:
                    nn.init.constant_(m.bias.to(device), 0)

    def forward(self, x):
        return self.layers(x)


def reproducibility(seed=9):
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True

