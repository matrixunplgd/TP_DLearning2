import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, random_split
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from tqdm import tqdm
import wandb

# 1. DATASET
class WineDataset(Dataset):
    def __init__(self, csv_path, scaler=None, fit_scaler=False):
        data = pd.read_csv(csv_path)
        self.features = data.iloc[:, 1:].values.astype(np.float32)
        self.labels = (data.iloc[:, 0].values - 1).astype(np.int64)
        
        if scaler:
            if fit_scaler:
                self.features = scaler.fit_transform(self.features)
            else:
                self.features = scaler.transform(self.features)
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        return torch.tensor(self.features[idx]), torch.tensor(self.labels[idx])

# 2. MODÈLE
class WineModel(nn.Module):
    def __init__(self, input_dim=13, hidden_dim=64, output_dim=3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, hidden_dim//2),
            nn.ReLU(),
            nn.Linear(hidden_dim//2, output_dim)
        )
    
    def forward(self, x):
        return self.net(x)

# 3. FONCTIONS D'ENTRAÎNEMENT ET ÉVALUATION
def train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss, correct, total = 0, 0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        out = model(x)
        loss = criterion(out, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        _, pred = torch.max(out, 1)
        correct += (pred == y).sum().item()
        total += y.size(0)
    return total_loss/len(loader), correct/total

def eval_epoch(model, loader, criterion, device):
    model.eval()
    total_loss, correct, total = 0, 0, 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            out = model(x)
            loss = criterion(out, y)
            total_loss += loss.item()
            _, pred = torch.max(out, 1)
            correct += (pred == y).sum().item()
            total += y.size(0)
    return total_loss/len(loader), correct/total

# 4. MAIN
def main():
    # Configuration
    config = {
        'batch_size': 32,
        'learning_rate': 0.001,
        'epochs': 50,
        'hidden_dim': 64,
        'dropout': 0.3,
        'optimizer': 'Adam',
        'device': 'cuda' if torch.cuda.is_available() else 'cpu'
    }
    
    # Initialisation wandb
    wandb.init(project="wine-classification", name=f"run_{config['hidden_dim']}_{config['learning_rate']}", config=config)
    
    # Chargement et normalisation (80/20 split)
    scaler = StandardScaler()
    full_dataset = WineDataset('wine.csv')
    train_size = int(0.8 * len(full_dataset))
    test_size = len(full_dataset) - train_size
    train_dataset, test_dataset = random_split(full_dataset, [train_size, test_size])
    
    # Normalisation sur les données d'entraînement uniquement
    train_features = torch.stack([full_dataset[i][0] for i in train_dataset.indices]).numpy()
    scaler.fit(train_features)
    
    # Recréer les datasets avec normalisation
    train_dataset = WineDataset('wine.csv', scaler, fit_scaler=False)
    test_dataset = WineDataset('wine.csv', scaler, fit_scaler=False)
    
    # Split final
    train_dataset, _ = random_split(train_dataset, [train_size, test_size])
    _, test_dataset = random_split(test_dataset, [train_size, test_size])
    
    # Dataloaders
    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=config['batch_size'], shuffle=False)
    
    # Modèle, loss, optimizer
    model = WineModel(hidden_dim=config['hidden_dim']).to(config['device'])
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config['learning_rate'])
    
    # Historiques
    train_losses, val_losses, train_accs, val_accs = [], [], [], []
    
    # Entraînement
    print("Début de l'entraînement...")
    for epoch in tqdm(range(config['epochs'])):
        train_loss, train_acc = train_epoch(model, train_loader, optimizer, criterion, config['device'])
        val_loss, val_acc = eval_epoch(model, test_loader, criterion, config['device'])
        
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accs.append(train_acc)
        val_accs.append(val_acc)
        
        # Log vers wandb
        wandb.log({
            'epoch': epoch + 1,
            'train_loss': train_loss,
            'train_accuracy': train_acc,
            'val_loss': val_loss,
            'val_accuracy': val_acc,
            'learning_rate': optimizer.param_groups[0]['lr']
        })
        
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}: Train Loss={train_loss:.4f}, Train Acc={train_acc:.4f} | Val Loss={val_loss:.4f}, Val Acc={val_acc:.4f}")
    
    # Visualisation
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    ax1.plot(train_losses, label='Train Loss')
    ax1.plot(val_losses, label='Validation Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True)
    
    ax2.plot(train_accs, label='Train Accuracy')
    ax2.plot(val_accs, label='Validation Accuracy')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    plt.savefig('training_curves.png')
    plt.show()
    
    # Sauvegarde des métriques finales dans wandb
    wandb.summary['best_val_accuracy'] = max(val_accs)
    wandb.summary['best_val_loss'] = min(val_losses)
    wandb.summary['final_test_accuracy'] = val_accs[-1]
    
    # Sauvegarde locale
    torch.save({
        'model_state': model.state_dict(),
        'config': config,
        'train_losses': train_losses,
        'val_losses': val_losses,
        'train_accs': train_accs,
        'val_accs': val_accs
    }, 'model.pth')
    
    print(f"\nTerminé! Meilleure accuracy validation: {max(val_accs):.4f}")
    print(f" Modèle sauvegardé dans 'model.pth'")
    print(f" Courbes sauvegardées dans 'training_curves.png'")
    print(f" Résultats visibles sur wandb: https://wandb.ai/")
    
    wandb.finish()

if __name__ == "__main__":
    main()