import torch, torchvision
import torch.nn as nn
from torch.nn.modules.activation import ReLU

# Model 1: Custom CNN(Tiny VGG)
class AiOrReal(nn.Module):
    def __init__(self, input_shape: int, hidden_units: int, output_shape: int):
        super().__init__()

        self.block_1 = nn.Sequential(
            nn.Conv2d(in_channels=input_shape,
                     out_channels=32,
                     kernel_size=3,
                     stride=1,
                     padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2,
                        stride=2)
        )
        self.block_2 = nn.Sequential(
            nn.Conv2d(in_channels=32,
                     out_channels=64,
                     kernel_size=3,
                     padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.block_3 = nn.Sequential(
            nn.Conv2d(in_channels = 64, out_channels = 128, kernel_size = 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(p=0.5),
            nn.Linear(in_features=128*28*28,
                     out_features=128),
            nn.ReLU(),
            nn.Linear(in_features=128,
                     out_features=output_shape)
        )

    def forward(self, x:torch.Tensor):
        x = self.block_1(x)
        x = self.block_2(x)
        x = self.block_3(x)
        x = self.classifier(x)

        return x

# Model 2: EfficientNet B0
def create_efficientnet_model(device: str = "cuda") -> nn.Module:
    """
    EfficientNet B0.
    Loads pretrained weights and modifies classifier.
    """
    weights = torchvision.models.EfficientNet_B0_Weights.DEFAULT
    model = torchvision.models.efficientnet_b0(weights=weights).to(device)
    
    # Freeze early layers, unfreeze last block
    for param in model.features[7:].parameters():
        param.requires_grad = True
    
    # Replace classifier
    model.classifier = torch.nn.Sequential(
        torch.nn.Linear(in_features=1280, out_features=512),
        torch.nn.ReLU(),
        torch.nn.Dropout(p=0.3),
        torch.nn.Linear(in_features=512, out_features=2)
    ).to(device)
    
    return model

# Model 3: ViT B-16
def create_vit_model(device: str = "cuda") -> nn.Module:
    """
    ViT B-16.
    Loads pretrained weights and modifies head.
    """
    weights = torchvision.models.ViT_B_16_Weights.DEFAULT
    model = torchvision.models.vit_b_16(weights=weights).to(device)
    
    # Freeze early layers, unfreeze last block
    for param in model.encoder.layers[11].parameters():
        param.requires_grad = True
    
    # Replace head
    model.heads = torch.nn.Sequential(
        torch.nn.Linear(in_features=768, 
                       out_features=2)
    ).to(device)
    
    return model

# Model 4: Hybrid EfficientNet + ViT
class AIHybrid(nn.Module):
    """
    Hybrid model.
    Fuses EfficientNet + ViT features.
    """
    def __init__(self, model_1, model_2):
        super().__init__()

        self.res_backbone = model_1.features
        self.res_pool = nn.AdaptiveAvgPool2d(1)

        self.vit_backbone = model_2

        # Combined Brain
        self.meta_classifier = nn.Sequential(
            nn.Linear(1280+768, 512),
            nn.ReLU(),
            nn.Dropout(p=0.4),
            nn.Linear(512, 2)
        )

    def forward(self, x):
        # 1. RES Path
        res_features = self.res_backbone(x)
        res_features = self.res_pool(res_features)
        res_features = torch.flatten(res_features, 1)

        # 2. ViT Path (Fixed)
        x_vit = self.vit_backbone._process_input(x)
        n = x_vit.shape[0]

        batch_class_token = self.vit_backbone.class_token.expand(n, -1, -1)
        x_vit = torch.cat([batch_class_token, x_vit], dim=1)

        vit_features = self.vit_backbone.encoder(x_vit)
        vit_features = vit_features[:, 0]

        # 3. Fusion Path
        combined_feat = torch.cat((res_features, vit_features), dim=1)

        return self.meta_classifier(combined_feat)
