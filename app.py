import torch
import torch.nn as nn
from torchvision import models
import albumentations as A
from albumentations.pytorch import ToTensorV2
from PIL import Image
import numpy as np
import gradio as gr

# 1. class labels

CLASS_NAMES = ["Healthy", "Powdery", "Rust"]


# 2. model definition
class PlantClassifier(nn.Module):
    def __init__(self, num_classes=3):
        super().__init__()
        self.model = models.resnet34(weights='IMAGENET1K_V1')
        for param in self.model.parameters():
            param.requires_grad = False
        self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)

    def forward(self, x):
        return self.model(x)


# 3. load checkpoint
def load_model():
    model = PlantClassifier(num_classes=3)
    checkpoint_path = "winner-epoch=01-val_acc=0.933-v1.ckpt"
    ckpt = torch.load(checkpoint_path, map_location="cpu")

    state_dict = ckpt["state_dict"]

    cleaned_state_dict = {}
    for k, v in state_dict.items():
        cleaned_state_dict[k.replace("model.", "")] = v

    model.model.load_state_dict(cleaned_state_dict)
    
    model.eval()
    return model

model = load_model()


# 4. preprocessing
transform = A.Compose([
    A.Resize(224, 224),
    A.Normalize(mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]),
    ToTensorV2()
])


# 5. prediction function
def predict(image):
    image = np.array(image.convert("RGB"))
    transformed = transform(image=image)["image"]
    x = transformed.unsqueeze(0)

    with torch.no_grad():
        outputs = model(x)
        probs = torch.softmax(outputs, dim=1)[0]

    confidences = {CLASS_NAMES[i]: float(probs[i]) for i in range(3)}
    predicted_class = CLASS_NAMES[int(torch.argmax(probs))]

    return predicted_class, confidences


# 6. gradio's user interface
demo = gr.Interface(
    fn=predict,
    inputs=gr.Image(type="pil"),
    outputs=[
        gr.Label(num_top_classes=1, label="Prediction"),
        gr.Label(label="Confidence Scores")
    ],
    title="Plant Disease Detection Demo",
    description="Upload a leaf image to classify it as Healthy, Powdery, or Rust."
)

demo.launch()
