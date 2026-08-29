
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset
import random
import copy
import time
import pickle
import math
import joblib



def save_complete_model(model, scaler_x, scaler_y, config, save_path='models/complete_model'):
    import os
    import json
    import pickle
    
    os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else '.', exist_ok=True)
    
    model_config = {
        'input_dim': config.get('input_dim', 5),
        'd_model': config.get('d_model', 256),
        'n_heads': config.get('n_heads', 4),
        'd_ff': config.get('d_ff', 48),
        'n_layers': config.get('n_layers', 4),
        'output_dim': config.get('output_dim', 2),
        'dropout': config.get('dropout', 0.1),
        'seq_len': config.get('seq_len', 15)
    }
    
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'model_config': model_config,
        'scaler_x': scaler_x,
        'scaler_y': scaler_y,
        'training_info': {
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'num_parameters': sum(p.numel() for p in model.parameters()),
            'model_class': model.__class__.__name__
        }
    }
    
    torch.save(checkpoint, f'{save_path}.pth')
    
    with open(f'{save_path}_config.json', 'w') as f:
        json.dump(model_config, f, indent=2)
    
    print(f"save: {save_path}.pth")
    print(f"model: {checkpoint['training_info']['num_parameters']:,}")
    print(f"setting: {model_config}")
    
    return save_path


class WindEstimationModel(nn.Module):
   
    
    def __init__(self, input_dim, d_model, n_heads, d_ff, n_layers, output_dim=2, dropout=0.1):
        super().__init__()
        
        assert input_dim == 5, "input_dim should be 5: [pitch, sin_yaw, cos_yaw, x, y]"
        

        self.output_layer = nn.Linear(d_model, output_dim)
        self.dropout = nn.Dropout(dropout)
 
        self.magnetic_encoder = ModalEncoder(
            input_dim=3, d_model=d_model, n_heads=n_heads, n_layers=n_layers, 
            d_ff=d_ff, dropout=dropout, modal_type='magnetic'
        )
        

        self.position_encoder = ModalEncoder(
            input_dim=2, d_model=d_model, n_heads=n_heads, n_layers=n_layers,
            d_ff=d_ff, dropout=dropout, modal_type='position'
        )
        
        self.cross_fusion = CrossAttentionFusion(
            d_model=d_model, n_heads=n_heads, dropout=dropout
        )
        
        self.output_proj = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, output_dim)
        )

    def forward(self, x, mask=None):
        batch_size, seq_len, _ = x.shape
        magnetic_data = x[:, :, :3]
        position_data = x[:, :, 3:] 
        magnetic_features, magnetic_attn_weights = self.magnetic_encoder(magnetic_data, mask)
        position_features, position_attn_weights = self.position_encoder(position_data, mask)
        fused_features, cross_attn_weights = self.cross_fusion(
            magnetic_features, position_features
        )
        output = self.output_proj(fused_features)
        attn_weights_list = [
            magnetic_attn_weights, 
            position_attn_weights, 
            cross_attn_weights
        ]
        
        return output, attn_weights_list

class ModalEncoder(nn.Module):
    
    def __init__(self, input_dim, d_model, n_heads, n_layers, d_ff, dropout=0.1, modal_type='magnetic'):
        super().__init__()
        self.modal_type = modal_type
        

        self.input_proj = nn.Linear(input_dim, d_model)
        

        if modal_type == 'position':
            self.pos_encoding = PositionalEncoding(d_model, dropout, max_len=1000)
        else:
            self.pos_encoding =  PositionalEncoding(d_model, dropout, max_len=1000)
     
        self.encoder_layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=n_heads,
                dim_feedforward=d_ff,
                dropout=dropout,
                batch_first=True
            )
            for _ in range(n_layers)
        ])
        
   
        self._init_modal_specific(modal_type)
    
    def _init_modal_specific(self, modal_type):
        if modal_type == 'magnetic':
           
            pass
        elif modal_type == 'position':
           
            pass
    
    def forward(self, x, mask=None):
        x = self.input_proj(x)
        x = self.pos_encoding(x)
        attn_weights_list = []
        for layer in self.encoder_layers:
            x, attn_weights = self._custom_forward(layer, x, src_mask=mask)
            attn_weights_list.append(attn_weights)
        sequence_embedding = x[:, -1, :]  
        return sequence_embedding, attn_weights_list
    
    def _custom_forward(self, layer, x, src_mask=None):
        
        x_out = layer(x, src_mask=src_mask)
        attn_weights = torch.ones(x.size(0), x.size(1), x.size(1))
        return x_out, attn_weights

class CrossAttentionFusion(nn.Module):
    
    
    def __init__(self, d_model, n_heads, dropout=0.1):
        super().__init__()
        
        self.mag_to_pos_attention = nn.MultiheadAttention(
            d_model, n_heads, dropout=dropout, batch_first=True
        )
        self.pos_to_mag_attention = nn.MultiheadAttention(
            d_model, n_heads, dropout=dropout, batch_first=True
        )
        self.fusion_gate = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model * 2),
            nn.Sigmoid()
        )
        self.layer_norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, magnetic_feat, position_feat):
        mag_expanded = magnetic_feat.unsqueeze(1)  # [batch, 1, d_model]
        pos_expanded = position_feat.unsqueeze(1)   # [batch, 1, d_model]
        pos_enhanced, attn_mag2pos = self.mag_to_pos_attention(
            query=pos_expanded,
            key=mag_expanded,
            value=mag_expanded
        )
        mag_enhanced, attn_pos2mag = self.pos_to_mag_attention(
            query=mag_expanded,
            key=pos_expanded, 
            value=pos_expanded
        )
        concat_features = torch.cat([pos_enhanced, mag_enhanced], dim=-1)
        gate_weights = self.fusion_gate(concat_features)
        gate1, gate2 = torch.chunk(gate_weights, 2, dim=-1)
        fused = gate1 * pos_enhanced + gate2 * mag_enhanced
        fused = self.layer_norm(fused)
        fused_features = fused.squeeze(1)  # [batch, d_model]
        
        return fused_features, (attn_mag2pos, attn_pos2mag)

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))  # [1, max_len, d_model]
    
    def forward(self, x):
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)

def train_model(model, train_loader, val_loader, num_epochs=10 , lr=0.0005):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10, factor=0.5)
    criterion = nn.MSELoss()

    train_losses = []
    val_losses = []

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            optimizer.zero_grad()
            outputs, _ = model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_loss += loss.item()
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                outputs, _ = model(batch_x)
                loss = criterion(outputs, batch_y)
                val_loss += loss.item()
        avg_train_loss = train_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)
        train_losses.append(avg_train_loss)
        val_losses.append(avg_val_loss)
        scheduler.step(avg_val_loss)
        if epoch % 1 == 0:
            print(f'Epoch [{epoch}/{num_epochs}], Train Loss: {avg_train_loss:.6f}, Val Loss: {avg_val_loss:.6f}')

    return train_losses, val_losses


def evaluate_model(model, test_loader, scaler_y):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.eval()

    predictions = []
    targets = []

    with torch.no_grad():
        for batch_x, batch_y in test_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            outputs, _ = model(batch_x)
            predictions.extend(outputs.cpu().numpy())
            targets.extend(batch_y.cpu().numpy())
    predictions = np.array(predictions)
    targets = np.array(targets)
    predictions_original = scaler_y.inverse_transform(predictions)
    targets_original = scaler_y.inverse_transform(targets)
    mse = np.mean((predictions_original - targets_original) ** 2)
    mae = np.mean(np.abs(predictions_original - targets_original))
    wind_x_mse = np.mean((predictions_original[:, 0] - targets_original[:, 0]) ** 2)
    wind_y_mse = np.mean((predictions_original[:, 1] - targets_original[:, 1]) ** 2)
    wind_x_mae = np.mean(np.abs(predictions_original[:, 0] - targets_original[:, 0]))
    wind_y_mae = np.mean(np.abs(predictions_original[:, 1] - targets_original[:, 1]))
    pred_wind_mag = np.sqrt(predictions_original[:, 0] ** 2 + predictions_original[:, 1] ** 2)
    true_wind_mag = np.sqrt(targets_original[:, 0] ** 2 + targets_original[:, 1] ** 2)
    pred_wind_dir = np.arctan2(predictions_original[:, 1], predictions_original[:, 0])
    true_wind_dir = np.arctan2(targets_original[:, 1], targets_original[:, 0])
    wind_mag_mae = np.mean(np.abs(pred_wind_mag - true_wind_mag))
    wind_dir_mae = np.mean(np.abs(np.arctan2(np.sin(pred_wind_dir - true_wind_dir),
                                             np.cos(pred_wind_dir - true_wind_dir))))
    print(f"MSE: {mse:.4f}, MAE: {mae:.4f}")
    print(f"X_component - MSE: {wind_x_mse:.4f}, MAE: {wind_x_mae:.4f}")
    print(f"Y_component - MSE: {wind_y_mse:.4f}, MAE: {wind_y_mae:.4f}")
    print(f"Speed MAE: {wind_mag_mae:.4f} m/s")
    print(f"Direction MAE: {np.degrees(wind_dir_mae):.2f}°")
    return predictions_original, targets_original


def visualize_attention(model, test_loader, scaler_x, scaler_y, num_examples=3):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.eval()
    with torch.no_grad():
        for batch_x, batch_y in test_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            outputs, attn_weights_list = model(batch_x)
            example_x = batch_x[0:1]  # (1, seq_len, input_dim)
            _, attn_weights = model(example_x)
            attn_weights = attn_weights_list[-1][0].mean(dim=0).cpu().numpy()  # (seq_len, seq_len)
            plt.figure(figsize=(8, 6))
            plt.imshow(attn_weights, cmap='hot', interpolation='nearest')
            plt.colorbar()
            plt.xlabel('Key Position')
            plt.ylabel('Query Position')
            plt.title('Attention Weights (Averaged over Heads)')
            plt.show()
            break

def build_windows_from_episodes(episodes, seq_len=15):
    data_temp = []
    targets_temp = []
    for episode in episodes:
        episode = np.asarray(episode, dtype=np.float32)
        if episode.ndim != 2 or episode.shape[1] < 7:
            raise ValueError(f"Episode should have shape [T, 7+], got {episode.shape}")
        for end in range(seq_len - 1, len(episode)):
            window = episode[end + 1 - seq_len:end + 1].copy()
            seq_x = window[:, :5].copy()
            target = window[-1, 5:7].copy()
            first_xy = seq_x[0, -2:].copy()
            seq_x[:, -2:] = first_xy - seq_x[:, -2:]
            data_temp.append(seq_x)
            targets_temp.append(target)
    return np.asarray(data_temp, dtype=np.float32), np.asarray(targets_temp, dtype=np.float32)

def build_windows_from_legacy_sequence(data_ini, seq_len=15):
    data_temp = []
    targets_temp = []
    for index in range(len(data_ini)):
        temp = data_ini[index][1:]
        for jdex in range((seq_len - 1), len(temp)):
            test11 = copy.deepcopy(temp[jdex + 1 - seq_len:(jdex + 1)])
            test_temp = np.array(test11, dtype=np.float32)
            test_temp = np.squeeze(test_temp, axis=1)
            seq_cpy1 = test_temp[:, 2:7]
            seq_cpy2 = test_temp[:, 7:9]
            first_xy = seq_cpy1[0, -2:].copy()
            seq_cpy1[:, -2:] = first_xy - seq_cpy1[:, -2:]
            data_temp.append(seq_cpy1)
            targets_temp.append(seq_cpy2[-1, :])
    return np.asarray(data_temp, dtype=np.float32), np.asarray(targets_temp, dtype=np.float32)

def main():
    with open('perception_sequences_static_policy_5d.pkl', 'rb') as f:
        data_ini = pickle.load(f)
    seq_len = 15
    if isinstance(data_ini, dict) and "episodes" in data_ini:
        data, targets = build_windows_from_episodes(data_ini["episodes"], seq_len=seq_len)
    elif isinstance(data_ini, dict) and "features" in data_ini and "targets" in data_ini:
        data = np.asarray(data_ini["features"], dtype=np.float32)
        targets = np.asarray(data_ini["targets"], dtype=np.float32)
    else:
        data, targets = build_windows_from_legacy_sequence(data_ini, seq_len=seq_len)
    print(f"data shape: {data.shape}")  
    print(f"targets shape: {targets.shape}")  
    print("Features: [pitch, sin_yaw, cos_yaw, relative_x, relative_y]")
    print("targets: [wind_x, wind_y]")
    batch_size, seq_len, feature_dim = data.shape
    data_2d = data.reshape(-1, feature_dim)
    targets_2d = targets.reshape(-1, 2)

    scaler_x = StandardScaler()
    scaler_y = StandardScaler()

    data_scaled = scaler_x.fit_transform(data_2d).reshape(batch_size, seq_len, feature_dim)
    targets_scaled = scaler_y.fit_transform(targets_2d)
    split_idx1 = int(0.7 * len(data))  
    split_idx2 = int(0.85 * len(data))  
    train_x = torch.tensor(data_scaled[:split_idx1])
    train_y = torch.tensor(targets_scaled[:split_idx1])
    val_x = torch.tensor(data_scaled[split_idx1:split_idx2])
    val_y = torch.tensor(targets_scaled[split_idx1:split_idx2])
    test_x = torch.tensor(data_scaled[split_idx2:])
    test_y = torch.tensor(targets_scaled[split_idx2:])
    train_dataset = TensorDataset(train_x, train_y)
    val_dataset = TensorDataset(val_x, val_y)
    test_dataset = TensorDataset(test_x, test_y)
    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=128)
    test_loader = DataLoader(test_dataset, batch_size=128)
    input_dim = 5  
    d_model = 256
    n_heads =  4
    d_ff = 48
    n_layers = 4
    output_dim = 2 
    hidden_dim = 64
    model = WindEstimationModel(input_dim, d_model, n_heads, d_ff, n_layers, output_dim)
    train_losses, val_losses = train_model(model, train_loader, val_loader, num_epochs=10)
    config = {
        'input_dim': input_dim,
        'd_model': d_model,
        'n_heads': n_heads,
        'd_ff': d_ff,
        'n_layers': n_layers,
        'output_dim': output_dim,
        'seq_len': seq_len
    }
    
    save_complete_model(
        model=model,
        scaler_x=scaler_x,
        scaler_y=scaler_y,
        config=config,
        save_path='wind_estimation_v1'
    )

    predictions, targets = evaluate_model(model, test_loader, scaler_y)
    import os
    output_dir = "supplementary_figures"
    os.makedirs(output_dir, exist_ok=True)
    plt.rcParams.update(
        {
            "font.family": "Arial",
            "font.size": 8,
            "axes.linewidth": 0.8,
            "xtick.major.width": 0.8,
            "ytick.major.width": 0.8,
            "xtick.direction": "out",
            "ytick.direction": "out",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    epochs = np.arange(1, len(train_losses) + 1)
    fig, ax = plt.subplots(figsize=(2.45, 1.85))
    ax.plot(
        epochs,
        train_losses,
        color="#2C7A7B",
        linewidth=1.6,
        label="Training",
    )
    ax.plot(
        epochs,
        val_losses,
        color="#D55E00",
        linewidth=1.6,
        label="Validation",
    )

    ax.set_xlabel("Epoch")
    ax.set_ylabel("MSE loss")
    ax.set_xlim(1, len(train_losses))
    ax.set_xticks([1, 5, len(train_losses)])
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(frameon=False, fontsize=7, loc="upper right")

    fig.tight_layout()

    for ext in ("pdf", "png"):
        fig.savefig(
            os.path.join(output_dir, f"supp_fig_s1a_training_curve.{ext}"),
            dpi=600,
            bbox_inches="tight",
        )

    plt.close(fig)



if __name__ == "__main__":
    main()














