import torch
from model import SimpleCNN

# Ladda den tränade modellen
model = SimpleCNN()
model.load_state_dict(torch.load("best_model.pth", weights_only=True))
model.eval()
print("Modellen laddad!")

# CIFAR-10 bilder är 32x32 pixlar med 3 färgkanaler (RGB)
# batch_size=1, channels=3, height=32, width=32
dummy_input = torch.randn(1, 3, 32, 32)

# Exportera till ONNX
torch.onnx.export(
    model,
    dummy_input,
    "model.onnx",
    input_names=["input"],
    output_names=["output"],
    dynamic_axes={"input": {0: "batch_size"}}
)

print("Modellen exporterad till model.onnx!")