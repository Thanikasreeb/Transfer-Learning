# Implementation-of-Transfer-Learning
## Aim
To Implement Transfer Learning for classification using VGG-19 architecture.
## Problem Statement and Dataset
Image classification from scratch requires large datasets and extensive training. Transfer Learning allows us to use pre-trained deep learning models (such as VGG-19 trained on ImageNet) and fine-tune them for our custom dataset, reducing computational cost and training time. In this experiment, we apply transfer learning using VGG-19 for a binary classification dataset, modifying the final layer to match the number of target classes.

## DESIGN STEPS
### STEP 1:
Import the required libraries, define dataset path and apply image transformations.

### STEP 2:
Load the pre-trained VGG-19 model and replace the final fully connected layer to match the number of classes in our dataset.

### STEP 3:
Define the loss function (BCEWithLogitsLoss) and optimizer (Adam) and train your model then check for results.

## PROGRAM
```python
# Load Pretrained Model and Modify for Transfer Learning
dataset_path = "./data/dataset/"
train_dataset = datasets.ImageFolder(root=f"{dataset_path}/train", transform=transform)
test_dataset = datasets.ImageFolder(root=f"{dataset_path}/test", transform=transform)
model = models.vgg19(pretrained=True)

# Modify the final fully connected layer to match the dataset classes
in_features = model.classifier[-1].in_features
model.classifier[-1] = nn.Linear(in_features, 1)

# Include the Loss function and optimizer
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.classifier[-1].parameters(), lr=0.001)

# Train the model
## Step 3: Train the Model
def train_model(model, train_loader,test_loader,num_epochs=10):
    train_losses = []
    val_losses = []
    model.train()
    for epoch in range(num_epochs):
        running_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device).float().unsqueeze(1) # Move data to device and adjust shape
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        train_losses.append(running_loss / len(train_loader))

        # Compute validation loss
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(device), labels.to(device).float().unsqueeze(1) # Move data to device and adjust shape
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item()

        val_losses.append(val_loss / len(test_loader))
        model.train()

        print(f'Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_losses[-1]:.4f}, Validation Loss: {val_losses[-1]:.4f}')

    # Plot training and validation loss
    print("Name:SUNIL KUMAR T")
    print("Register Number:212223240164")
    plt.figure(figsize=(8, 6))
    plt.plot(range(1, num_epochs + 1), train_losses, label='Train Loss', marker='o')
    plt.plot(range(1, num_epochs + 1), val_losses, label='Validation Loss', marker='s')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.show()
train_model(model, train_loader,test_loader,num_epochs=10)
```

## OUTPUT
### Training Loss, Validation Loss Vs Iteration Plot
<img width="932" height="662" alt="image" src="https://github.com/user-attachments/assets/9abbc6f6-0404-4498-b98c-70e121a14305" />

### Confusion Matrix
<img width="782" height="638" alt="image" src="https://github.com/user-attachments/assets/ac98ef48-46b6-4f42-b929-d6eabac7204b" />


### Classification Report
<img width="477" height="202" alt="image" src="https://github.com/user-attachments/assets/022d1557-ba4f-4b1d-8af9-f3aa0c739ceb" />


### New Sample Prediction
<img width="590" height="468" alt="image" src="https://github.com/user-attachments/assets/b38d3af9-89ab-483e-a584-5a9b47820af1" />

<img width="487" height="452" alt="image" src="https://github.com/user-attachments/assets/310438d0-a07c-42c0-9826-af569a25d061" />


## RESULT
Therefore, transfer learning using the VGG-19 architecture was successfully implemented for classification.

