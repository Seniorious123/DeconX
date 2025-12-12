import os
import torch
import numpy as np
from tqdm import tqdm
from torch.optim import Adam
import torch.nn.functional as F

from .utils import showloss_plot, device

def predict_ae(ae_model, data_loader):
    """
    Predicts cell fractions using the encoder part of an AutoEncoder model.
    """
    ae_model.eval()
    predictions = []
    with torch.no_grad():
        for data in data_loader:
            if isinstance(data, (list, tuple)):
                data = data[0]
            data = data.to(device)
            _, cell_frac, _ = ae_model(data)
            predictions.append(cell_frac.cpu().numpy())
    return np.vstack(predictions)

def training_stage(model, train_loader, optimizer, epochs=128, frac_lambda=1000, 
                   sig_lambda=1.0, reference_sig=None, use_mse=False):
    """
    The core training loop for the AutoEncoder model.
    Includes logic for a composite loss function with signature preservation.
    """
    model.train()
    loss_history = {'frac': [], 'recon': [], 'sig': [], 'total': []}

    if reference_sig is not None:
        reference_sig_tensor = torch.from_numpy(reference_sig).float().to(device)
        num_known_cells = reference_sig.shape[0]

    for i in tqdm(range(epochs), desc="Training Epochs"):
        epoch_losses = {'frac': [], 'recon': [], 'sig': [], 'total': []}
        
        for data, label in train_loader:
            data = data.to(device)
            label = label.to(device)
            optimizer.zero_grad()
            
            x_recon, cell_frac, current_sig = model(data)
            
            loss_func = F.mse_loss if use_mse else F.l1_loss
            recon_loss = loss_func(x_recon, data)
            frac_loss = loss_func(cell_frac, label)
            
            sig_loss = torch.tensor(0.0).to(device)
            if reference_sig is not None:
                current_known_sig = current_sig[:num_known_cells, :]
                sig_loss = F.mse_loss(current_known_sig, reference_sig_tensor)




            total_loss = recon_loss + frac_lambda * frac_loss + sig_lambda * sig_loss 
            
            total_loss.backward()
            optimizer.step()

            epoch_losses['recon'].append(recon_loss.item())
            epoch_losses['frac'].append(frac_loss.item())
            epoch_losses['sig'].append(sig_loss.item())
            epoch_losses['total'].append(total_loss.item())

        # Print average losses every 10 epochs
        if i % 10 == 0 or i == epochs - 1:
            avg_recon = np.mean(epoch_losses['recon'])
            avg_frac = np.mean(epoch_losses['frac'])
            avg_sig = np.mean(epoch_losses['sig'])
            
            print(f"Epoch {i}:")
            print(f"  Avg Recon Loss: {avg_recon:.6f}")
            print(f"  Avg Frac Loss: {avg_frac:.6f} (Weighted: {frac_lambda * avg_frac:.6f})")
            print(f"  Avg Sig Loss: {avg_sig:.6f} (Weighted: {sig_lambda * avg_sig:.6f})")

            

        for key in loss_history:
            loss_history[key].append(np.mean(epoch_losses[key]))
            
    return model, loss_history

def train_model(model, train_loader, epochs=128, act_lr=1e-4, frac_lambda=1000, 
                sig_lambda=1.0, reference_sig=None, use_mse=False, 
                output_dir='', output_name=''):
    """
    A wrapper function to train the model and save loss plots and the final model.
    """
    optimizer = Adam(model.parameters(), lr=act_lr)
    
    model, loss_history = training_stage(
        model, train_loader, optimizer, epochs=epochs, 
        frac_lambda=frac_lambda, sig_lambda=sig_lambda, 
        reference_sig=reference_sig, use_mse=use_mse
    )
    
    if output_dir and output_name:
        for loss_type, values in loss_history.items():
            if loss_type == 'sig' and reference_sig is None:
                continue
            showloss_plot(values, 
                          os.path.join(output_dir, f"{output_name}{loss_type}_loss.png"), 
                          y_axis=f"{loss_type.capitalize()} ")
    
        model_path = os.path.join(output_dir, output_name + "model.pth")
        torch.save(model, model_path)
        print(f"Model saved to {model_path}")
    
    return model
