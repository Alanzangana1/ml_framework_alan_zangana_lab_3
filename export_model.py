import torch
from model import SimpleCNN


model = SimpleCNN()
model.load_state_dict(torch.load("best_model.pth", weights_only=True))
model.eval()
print("Modellen laddad!")


dummy_input = torch.randn(1, 3, 32, 32)


torch.onnx.export(
    model,
    dummy_input,
    "model.onnx",
    input_names=["input"],
    output_names=["output"],
    dynamic_axes={"input": {0: "batch_size"}}
)

print("Modellen exporterad till model.onnx!")