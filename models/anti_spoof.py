import torch
import torch.nn as nn
from torchvision import models
from torchvision.models import MobileNet_V2_Weights

#  Model
class AntiSpoofNet(nn.Module):
    """
    MobileNetV2 fine-tuned cho bài toán phân loại Real (1) / Fake (0).

    Args:
        num_classes  : 2 (Real / Fake)
        pretrained   : Dùng ImageNet weights hay không
        dropout_rate : Dropout trước lớp cuối để tránh overfit
    """

    def __init__(
        self,
        num_classes: int = 2,
        pretrained: bool = True,
        dropout_rate: float = 0.4,
    ):
        super().__init__()

        # ── Backbone ─────────────────────────────────────────────────────
        weights = MobileNet_V2_Weights.IMAGENET1K_V1 if pretrained else None
        backbone = models.mobilenet_v2(weights=weights)

        # Giữ lại feature extractor, bỏ classifier gốc
        self.features = backbone.features          # Output: (B, 1280, 4, 4) với input 112x112

        # Global Average Pooling: (B, 1280, H, W) → (B, 1280)
        self.gap = nn.AdaptiveAvgPool2d(1)

        # ── Classifier head mới ──────────────────────────────────────────
        # 1280 → 256 → num_classes
        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout_rate),
            nn.Linear(1280, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout_rate * 0.5),
            nn.Linear(256, num_classes),
        )

        # Khởi tạo trọng số lớp Linear mới
        self._init_classifier()

    def _init_classifier(self):
        for m in self.classifier.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode="fan_out")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)          # Feature extraction
        x = self.gap(x)               # Global Average Pool
        x = x.flatten(1)              # Flatten → (B, 1280)
        x = self.classifier(x)        # Classification → (B, 2)
        return x

#  Freeze / Unfreeze helpers
def freeze_backbone(model: AntiSpoofNet):
    """Phase 1: Đóng băng toàn bộ backbone, chỉ train classifier."""
    for param in model.features.parameters():
        param.requires_grad = False
    for param in model.classifier.parameters():
        param.requires_grad = True

    n_frozen  = sum(p.numel() for p in model.features.parameters())
    n_trainable = sum(p.numel() for p in model.classifier.parameters())
    print(f"[freeze_backbone] Frozen: {n_frozen:,} params | "
          f"Trainable: {n_trainable:,} params")


def unfreeze_last_n_layers(model: AntiSpoofNet, n: int = 5):
    """
    Phase 2: Mở băng n InvertedResidual block cuối của backbone.
    MobileNetV2.features có 19 block (index 0–18).
    Thường unfreeze block 14–18 (5 block cuối) là đủ.
    """
    # Đầu tiên freeze tất cả
    for param in model.features.parameters():
        param.requires_grad = False

    # Sau đó unfreeze n block cuối
    total_blocks = len(model.features)          # = 19
    unfreeze_from = max(0, total_blocks - n)

    for i in range(unfreeze_from, total_blocks):
        for param in model.features[i].parameters():
            param.requires_grad = True

    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[unfreeze_last_{n}] Trainable params: {n_trainable:,} "
          f"(backbone block {unfreeze_from}–{total_blocks-1} + classifier)")
    
#  Test
if __name__ == "__main__":
    model = AntiSpoofNet(pretrained=True)

    print("\n=== Phase 1: Freeze backbone ===")
    freeze_backbone(model)

    print("\n=== Phase 2: Unfreeze 5 block cuối ===")
    unfreeze_last_n_layers(model, n=5)

    # Forward pass test
    x = torch.randn(4, 3, 112, 112)
    logits = model(x)
    print(f"\nForward pass OK | Input: {x.shape} → Output: {logits.shape}")
    print(f"Logits sample: {logits[0].detach().numpy()}")
