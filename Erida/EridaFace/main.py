from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
from safetensors.torch import load_file
from torchvision import transforms
from mtcnn import MTCNN
import numpy as np
import matplotlib.pyplot as plt

class MSAG(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(MSAG, self).__init__()
        self.branch1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )
        self.branch2 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=5, padding=2),
            nn.ReLU(inplace=True)
        )
        self.branch3 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=7, padding=3),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        branch1 = self.branch1(x)
        branch2 = self.branch2(x)
        branch3 = self.branch3(x)
        return branch1, branch2, branch3

class AttentionModule(nn.Module):
    def __init__(self, in_channels):
        super(AttentionModule, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, in_channels // 2, kernel_size=1)
        self.conv2 = nn.Conv2d(in_channels // 2, 1, kernel_size=1)

    def forward(self, x):
        attn = F.relu(self.conv1(x))
        attn = torch.sigmoid(self.conv2(attn))
        return x * attn

class FusionBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(FusionBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)

    def forward(self, x1, x2, x3):
        x = torch.cat([x1, x2, x3], dim=1)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        return x

class Decoder(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Decoder, self).__init__()
        self.upconv1 = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
        self.conv1 = nn.Conv2d(in_channels // 2, out_channels, kernel_size=3, padding=1)

    def forward(self, x):
        x = F.relu(self.upconv1(x))
        x = self.conv1(x)
        return x

class MSAGNet(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(MSAGNet, self).__init__()
        self.feature_extractor = MSAG(in_channels, 64)
        self.attention = AttentionModule(64)
        self.fusion = FusionBlock(192, 128)
        self.decoder = Decoder(128, out_channels)

    def forward(self, x):
        branch1, branch2, branch3 = self.feature_extractor(x)
        branch1 = self.attention(branch1)
        branch2 = self.attention(branch2)
        branch3 = self.attention(branch3)
        fused = self.fusion(branch1, branch2, branch3)
        output = self.decoder(fused)
        output = F.interpolate(output, size=x.shape[2:], mode='bilinear', align_corners=False)
        output = torch.sigmoid(output)
        return output

model_path = r'Z:\erida-pulo\MSAG-Net\model.safetensors'
model_weights = load_file(model_path)

model = MSAGNet(in_channels=3, out_channels=1)
model.load_state_dict(model_weights)
model.eval()
face_detector = MTCNN()

def preprocess_image(image_path, target_size=(256, 256)):
    image = Image.open(image_path).convert('RGB')
    image = image.resize(target_size)
    image = np.array(image) / 255.0
    image = torch.tensor(image).permute(2, 0, 1).float().unsqueeze(0)
    return image

def visualize_results(image, mask, original_image_path):
    image = image.squeeze().permute(1, 2, 0).numpy()
    mask = mask.squeeze().numpy()
    mask = (mask > 0.5).astype(np.uint8)

    plt.figure(figsize=(12, 6))
    plt.subplot(1, 3, 1)
    plt.title('Оригинал')
    plt.imshow(Image.open(original_image_path))
    plt.axis('off')

    plt.subplot(1, 3, 2)
    plt.title('С личиком')
    plt.imshow(image)
    plt.axis('off')

    plt.show()

test_image_path = r'Z:\erida-pulo\MSAG-Net\test2.jpg'
input_image = preprocess_image(test_image_path)

pil_image = transforms.ToPILImage()(input_image.squeeze())
np_image = np.array(pil_image)

faces = face_detector.detect_faces([np_image])
if not faces:
    print("No faces detected.")
else:
    face_images = []
    for face in faces[0]:
        x, y, w, h = face['box']
        face_image = input_image[:, :, y:y+h, x:x+w]
        face_image = F.interpolate(face_image, size=(256, 256))
        face_images.append(face_image)

    if face_images:
        face_images = torch.cat(face_images)
        with torch.no_grad():
            output_mask = model(face_images)

        visualize_results(face_images, output_mask, test_image_path)

