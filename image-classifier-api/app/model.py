import torch
import torchvision.transforms as transforms
from torchvision.models import resnet50, ResNet50_Weights
from PIL import Image
import urllib.request
import json
import io

LABELS_URL = "https://raw.githubusercontent.com/anishathalye/imagenet-simple-labels/master/imagenet-simple-labels.json"

with urllib.request.urlopen(LABELS_URL) as response:
    LABELS = json.loads(response.read().decode())

print("Loading ResNet50...")
model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
model.eval()  # inference mode — disables dropout, batchnorm behaves differently
print("Model ready.")

transform = transforms.Compose([
    transforms.Resize(256),                          # resize shortest edge to 256
    transforms.CenterCrop(224),                      # crop to 224x224 (ResNet input size)
    transforms.ToTensor(),                           # convert PIL image to tensor [0,1]
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],                  # ImageNet mean per channel
        std=[0.229, 0.224, 0.225]                    # ImageNet std per channel
    )
])

def predict(image_bytes: bytes, top_k: int = 5) -> list[dict]:
    # Load image from raw bytes
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

    # Apply preprocessing — output shape: [1, 3, 224, 224]
    tensor = transform(image).unsqueeze(0)

    # Forward pass — no gradient needed at inference
    with torch.no_grad():
        outputs = model(tensor)

    # Convert raw logits to probabilities
    probabilities = torch.nn.functional.softmax(outputs[0], dim=0)

    # Get top-k predictions
    top_probs, top_indices = torch.topk(probabilities, top_k)

    results = []
    for prob, idx in zip(top_probs.tolist(), top_indices.tolist()):
        results.append({
            "label": LABELS[idx],
            "confidence": round(prob, 4)
        })

    return results