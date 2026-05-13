import torch 
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

print(torch.__version__)

class CustomDataset(Dataset):
    def __init__(self):
        self.data = torch.randn(size=(100, 3))  # 100 samples, 3 features
        self.labels = torch.randint(low=0, high=2, size=(100,))  # Binary labels
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        data = self.data[idx]
        label = self.labels[idx]
        return {
            "input": data,   # J'ai enlevé les ":" (c'était une erreur)
            "label": label   # J'ai enlevé les ":"
        }
        
    def get_dataloader(self, batch_size=16):
        return DataLoader(dataset=self, batch_size=batch_size, shuffle=True)

# Modèle en dehors de la classe CustomDataset (indentation corrigée)
class LinearModel(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.linear = nn.Linear(in_features=in_dim, out_features=out_dim)
        
    def forward(self, input):
        output = self.linear(input)
        return output

def train_epoch(model, dataloader, optimizer, criterion):
    model.train()  # Mode entraînement
    total_loss = 0
    
    for batch in dataloader:
        # Récupérer les données du batch
        inputs = batch["input"]   # Forme: (batch_size, 3)
        labels = batch["label"]   # Forme: (batch_size,)
        
        # 1. Reset des gradients
        optimizer.zero_grad()
        
        # 2. Forward pass (prédiction)
        outputs = model(inputs)    # Forme: (batch_size, 12)
        
        # 3. Calcul de la perte
        # Attention: outputs est (20,12) mais labels est (20,)
        # Pour une classification binaire, on veut souvent (20,1) ou utiliser CrossEntropyLoss
        loss = criterion(outputs, labels)
        
        # 4. Backward pass (calcul des gradients)
        loss.backward()
        
        # 5. Mise à jour des poids
        optimizer.step()
        
        loss.append(loss)

        total_loss += loss.item()
    
    return total_loss / len(dataloader)  # Perte moyenne sur l'époque

def main():
    # Paramètres
    BATCH_SIZE = 20
    IN_DIM = 3      # 3 features en entrée
    OUT_DIM = 2    # 12 neurones en sortie (ça peut être trop pour 2 classes)
    LEARNING_RATE = 0.05
    EPOCHS = 1000
    
    # Création du dataset et dataloader
    mydataset = CustomDataset()
    mydataloader = mydataset.get_dataloader(batch_size=BATCH_SIZE)
    
    # Création du modèle
    model = LinearModel(in_dim=IN_DIM, out_dim=OUT_DIM)
    
    # Définition de la fonction de perte (loss)
    # CrossEntropyLoss pour classification multi-classes
    criterion = nn.CrossEntropyLoss()
    
    # Définition de l'optimiseur
    optimizer = torch.optim.SGD(model.parameters(), lr=LEARNING_RATE)
    
    # Boucle d'entraînement
    for epoch in range(EPOCHS):
        loss = train_epoch(model, mydataloader, optimizer, criterion)
        print(f"Epoch {epoch+1}/{EPOCHS}, Loss: {loss:.4f}")
    
    print("Entraînement terminé !")

if __name__ == "__main__":
    main()