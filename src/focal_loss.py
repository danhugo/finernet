import torch
from torchvision.ops import sigmoid_focal_loss

# this is based on bce with logits loss
# alpha: balance pos and neg. if pos is rare, set to 0.25 or 0.1. if blance can set to 0.5 or 0.75
# gamma: to balance easy and hard example: start with 0.5 or 1 then increase
    
# Example input
num_classes = 1  # Binary classification
batch_size = 4

# Example logits (or probabilities)
pred_logits = torch.randn(batch_size, num_classes)  # Assuming logits
pred_probs = torch.sigmoid(pred_logits)  # Convert logits to probabilities using sigmoid
print("Predicted Probabilities:")
print(pred_probs)

# Example ground truth labels
target = torch.randint(0, 2, (batch_size,), dtype=torch.float)
print("Ground Truth Labels:")
print(target)

# Get fake predictions from our "model" and compare
logits = torch.randn(batch_size, num_classes)
loss = sigmoid_focal_loss(pred_probs.squeeze(), target, reduction='mean')
print("Focal Loss2:", loss)