from torch_geometric.loader import DataLoader
from sklearn.model_selection import train_test_split
import torch.optim as optim
# Split dataset into train, validation, and test sets
train_data, temp_data = train_test_split(data_list, test_size=0.2, random_state=42)
val_data, test_data = train_test_split(temp_data, test_size=0.5, random_state=42)

train_loader = DataLoader(train_data, batch_size=32, shuffle=True, drop_last=True)
val_loader = DataLoader(val_data, batch_size=32, shuffle=False, drop_last=True)
test_loader = DataLoader(test_data, batch_size=32, shuffle=False, drop_last=True)

# Initialize model and optimizer
model = GCN()
optimizer = optim.Adam(model.parameters(), lr=0.01)

# Initialize variables for storing best validation accuracy and corresponding model state
best_val_acc = 0.0
best_model_state = None

# Training loop
for epoch in range(10):
    model.train()
    train_loss = 0
    train_correct = 0
    
    # Training
    for batch in train_loader:
        optimizer.zero_grad()
        out = model(batch)
        print(f"Batch Y shape: {batch.y.shape}, Output shape: {out.shape}")
        loss = F.nll_loss(out, batch.y)
        loss.backward()
        optimizer.step()
        
        train_loss += loss.item()
        pred = out.argmax(dim=1)
        train_correct += int((pred == batch.y).sum())
    
    train_loss /= len(train_loader.dataset)
    train_acc = train_correct / len(train_loader.dataset)
    
    # Validation
    model.eval()
    val_loss = 0
    val_correct = 0
    with torch.no_grad():
        for batch in val_loader:
            out = model(batch)
            loss = F.nll_loss(out, batch.y)
            
            val_loss += loss.item()
            pred = out.argmax(dim=1)
            val_correct += int((pred == batch.y).sum())
    
    val_loss /= len(val_loader.dataset)
    val_acc = val_correct / len(val_loader.dataset)
    
    print(f"Epoch {epoch+1}, Train Loss: {round(train_loss, 4)}, Train Acc: {round(train_acc, 4)}, Val Loss: {round(val_loss, 4)}, Val Acc: {round(val_acc, 4)}")
    
    # Save the model if validation accuracy improves
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        best_model_state = model.state_dict()
        torch.save(best_model_state, "best_model.pth")

# Load the best model
model.load_state_dict(torch.load("best_model.pth"))

# Test the model
model.eval()
test_correct = 0
with torch.no_grad():
    for batch in test_loader:
        out = model(batch)
        pred = out.argmax(dim=1)
        test_correct += int((pred == batch.y).sum())

test_acc = test_correct / len(test_loader.dataset)
print(f"Test Acc: {round(test_acc, 4)}")
