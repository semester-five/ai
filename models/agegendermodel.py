import torch
import torch.nn as nn
from torchvision.models import mobilenet_v3_large, MobileNet_V3_Large_Weights

class AgeGenderMobileNetV3(nn.Module):
    """
    MobileNetV3-Large fine-tuned cho bài toán:
    - Tuổi: hồi quy (regression) – đầu ra 1 giá trị (range: 0-120)
    - Giới tính: phân loại nhị phân – đầu ra 2 logits
      * 0: Nam (male)
      * 1: Nữ (female)
    """
    def __init__(self, pretrained=True):
        super(AgeGenderMobileNetV3, self).__init__()
        
        # 1. Tải backbone MobileNetV3-Large với pretrained weights
        if pretrained:
            weights = MobileNet_V3_Large_Weights.DEFAULT
            self.backbone = mobilenet_v3_large(weights=weights)
        else:
            self.backbone = mobilenet_v3_large(weights=None)
        
        # 2. Lấy số đặc trưng đầu ra của backbone (1280 đối với large)
        num_features = self.backbone.classifier[0].in_features  # thường là 1280
        
        # 3. Loại bỏ classifier gốc của MobileNetV3
        self.backbone.classifier = nn.Identity()
        
        # 4. Xây dựng các đầu ra riêng cho từng tác vụ
        #    Dùng 2 lớp Linear với BatchNorm1d và Dropout
        self.age_head = nn.Sequential(
            nn.Linear(num_features, 128),
            nn.BatchNorm1d(128),
            nn.Hardswish(inplace=True),
            nn.Dropout(p=0.2),
            nn.Linear(128, 1)   # regression output
        )
        
        self.gender_head = nn.Sequential(
            nn.Linear(num_features, 128),
            nn.BatchNorm1d(128),
            nn.Hardswish(inplace=True),
            nn.Dropout(p=0.2),
            nn.Linear(128, 2)   # 2 classes: female, male
        )

    def freeze_backbone(self):
        """
        Đóng băng toàn bộ backbone – chỉ train các head.
        Dùng trong vài epoch đầu để tránh catastrophic forgetting.
        """
        for param in self.backbone.parameters():
            param.requires_grad = False

    def unfreeze_backbone(self):
        """
        Mở băng toàn bộ backbone để fine-tune end-to-end.
        Gọi sau khi các head đã hội tụ ổn định.
        """
        for param in self.backbone.parameters():
            param.requires_grad = True

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): batch ảnh đầu vào, shape (B, 3, H, W)
        Returns:
            age (torch.Tensor): shape (B, 1) – giá trị tuổi dự đoán (0-120)
            gender (torch.Tensor): shape (B, 2) – logits cho 2 lớp (Nam/Nữ)
        """
        features = self.backbone(x)          # shape (B, num_features)
        age = self.age_head(features)        # (B, 1) – raw output
        age = torch.clamp(age, min=0, max=120)  # Clamp vào range [0, 120]
        gender = self.gender_head(features)  # (B, 2)
        return age, gender


# ============================================================================
# Loss Functions và Utilities cho training
# ============================================================================

class AgeGenderLosses:
    """
    Container cho loss functions của age & gender tasks.
    Có thể dùng cho weighted combination multi-task learning.

    Lý do chọn loss weight mặc định (age=0.1, gender=1.0):
      - HuberLoss trên tuổi thường có magnitude ~5–15 (đơn vị năm),
        trong khi CrossEntropyLoss cho gender thường ~0.3–0.7.
      - Nếu để weight bằng nhau, age loss sẽ áp đảo gradient và
        khiến model bỏ qua task gender.
      - age_loss_weight=0.1 giúp cân bằng đóng góp của hai task,
        có thể điều chỉnh lại sau khi quan sát loss curve thực tế.
    """
    def __init__(self, age_loss_weight=0.1, gender_loss_weight=1.0):
        """
        Args:
            age_loss_weight (float): Weight cho age regression loss.
                                     Mặc định 0.1 để cân bằng với gender loss.
            gender_loss_weight (float): Weight cho gender classification loss.
        """
        # HuberLoss (smooth L1) ít nhạy cảm với outlier hơn MSELoss.
        # delta=5.0: sai số < 5 tuổi dùng L2, sai số > 5 tuổi dùng L1.
        self.age_criterion = nn.HuberLoss(delta=5.0)
        self.gender_criterion = nn.CrossEntropyLoss()   # cho 2 classes
        self.age_loss_weight = age_loss_weight
        self.gender_loss_weight = gender_loss_weight
    
    def compute_loss(self, age_pred, age_target, gender_pred, gender_target):
        """
        Tính total loss với weighted combination.
        
        Args:
            age_pred (torch.Tensor): shape (B, 1) – dự đoán tuổi
            age_target (torch.Tensor): shape (B,) hoặc (B, 1) – tuổi thực (float)
            gender_pred (torch.Tensor): shape (B, 2) – logits cho gender
            gender_target (torch.Tensor): shape (B,) – gender label (0 or 1, long)
        
        Returns:
            total_loss (torch.Tensor): scalar
            loss_dict (dict): {'age_loss': ..., 'gender_loss': ..., 'total': ...}
        """
        # Reshape để match
        age_pred = age_pred.squeeze(-1) if age_pred.dim() == 2 else age_pred
        age_target = age_target.float()
        
        # Tính loss từng task
        age_loss = self.age_criterion(age_pred, age_target)
        gender_loss = self.gender_criterion(gender_pred, gender_target)
        
        # Weighted combination
        total_loss = (self.age_loss_weight * age_loss + 
                      self.gender_loss_weight * gender_loss)
        
        loss_dict = {
            'age_loss': age_loss.item(),
            'gender_loss': gender_loss.item(),
            'total': total_loss.item()
        }
        
        return total_loss, loss_dict

class SafeAgeGenderLosses(AgeGenderLosses):
    """Wrapper around AgeGenderLosses with better error handling for Colab"""

    def compute_loss_safe(self, age_pred, age_target, gender_pred, gender_target):
        """Compute loss with error handling và CPU fallback"""
        try:
            # Ensure all tensors are on same device
            if age_pred.device != age_target.device:
                age_target = age_target.to(age_pred.device)
            if gender_pred.device != gender_target.device:
                gender_target = gender_target.to(gender_pred.device)

            # Clamp age predictions to valid range [0, 120]
            age_pred_clamped = torch.clamp(age_pred.squeeze(), min=0, max=120)

            # Ensure data types are correct
            age_target = age_target.float()
            gender_target = gender_target.long()

            # Split để compute riêng từng loss
            age_pred_flat = age_pred_clamped if age_pred_clamped.dim() == 1 else age_pred_clamped.squeeze()
            age_loss = self.age_criterion(age_pred_flat, age_target)
            gender_loss = self.gender_criterion(gender_pred, gender_target)

            # Check for NaN/Inf
            if torch.isnan(age_loss) or torch.isinf(age_loss):
                print(f"⚠️ Warning: Age loss is {age_loss.item()}, using fallback")
                age_loss = torch.tensor(0.0, device=age_pred.device, requires_grad=True)

            if torch.isnan(gender_loss) or torch.isinf(gender_loss):
                print(f"⚠️ Warning: Gender loss is {gender_loss.item()}, using fallback")
                gender_loss = torch.tensor(0.0, device=gender_pred.device, requires_grad=True)

            total_loss = (self.age_loss_weight * age_loss +
                         self.gender_loss_weight * gender_loss)

            loss_dict = {
                'age_loss': age_loss.item(),
                'gender_loss': gender_loss.item(),
                'total': total_loss.item()
            }

            return total_loss, loss_dict

        except Exception as e:
            print(f"❌ Error in loss computation: {e}")
            raise

def get_optimizer(model, lr=1e-3):
    """
    Tạo optimizer (Adam) cho model.
    Chỉ optimize các params có requires_grad=True,
    nên hoạt động đúng khi backbone đang bị freeze.
    """
    return torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=lr
    )


def get_scheduler(optimizer, total_epochs, warmup_epochs=3):
    """
    Tạo learning rate scheduler với Linear Warmup + CosineAnnealing.

    Cơ chế:
      - Warmup (warmup_epochs epoch đầu): LR tăng tuyến tính từ lr*0.1 → lr.
        Giúp model ổn định trước khi fine-tune với LR đầy đủ.
      - Cosine Annealing (các epoch còn lại): LR giảm dần theo hàm cosine → 0.
        Tránh bị kẹt ở local minima cuối quá trình training.

    Args:
        optimizer: optimizer đã khởi tạo.
        total_epochs (int): Tổng số epoch training.
        warmup_epochs (int): Số epoch warmup. Mặc định 3.

    Returns:
        scheduler (SequentialLR)
    """
    from torch.optim.lr_scheduler import LinearLR, CosineAnnealingLR, SequentialLR

    # Warmup: LR tăng từ 10% lên 100% trong warmup_epochs epoch
    warmup_scheduler = LinearLR(
        optimizer,
        start_factor=0.1,
        end_factor=1.0,
        total_iters=warmup_epochs
    )

    # Cosine decay sau warmup
    cosine_scheduler = CosineAnnealingLR(
        optimizer,
        T_max=total_epochs - warmup_epochs
    )

    return SequentialLR(
        optimizer,
        schedulers=[warmup_scheduler, cosine_scheduler],
        milestones=[warmup_epochs]
    )