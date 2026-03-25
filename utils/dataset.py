import os
from pathlib import Path
from PIL import Image

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

def _label_from_name(filename: str) -> int:
    name = filename.lower()
    if name.endswith("_real.jpg"):
        return 1
    if name.endswith("_fake.jpg"):
        return 0
    return -1  # Bỏ qua file không hợp lệ


class CasiaFASDDataset(Dataset):
    """
    Dataset cho CASIA-FASD (chỉ dùng ảnh color, không cần depth).

    Cấu trúc thư mục sau khi giải nén:
        root/
        ├── train_img/
        │   └── train_img/
        │       └── color/   ← *_real.jpg, *_fake.jpg
        └── test_img/
            └── test_img/
                └── color/   ← *_real.jpg, *_fake.jpg

    Args:
        root_dir  : đường dẫn đến thư mục gốc (chứa train_img/ và test_img/)
        split     : "train" hoặc "test"
        transform : torchvision transforms (nếu None dùng transform mặc định)
    """

    # Đường dẫn con tới thư mục color (có thể điều chỉnh nếu cấu trúc khác)
    SPLIT_MAP = {
        "train": os.path.join("train_img", "color"),
        "test":  os.path.join("test_img",   "color"),
    }

    def __init__(self, root_dir: str, split: str = "train", transform=None):
        assert split in self.SPLIT_MAP, f"split phải là 'train' hoặc 'test', nhận: {split}"

        self.color_dir = Path(root_dir) / self.SPLIT_MAP[split]
        assert self.color_dir.exists(), (
            f"Không tìm thấy thư mục color:\n  {self.color_dir}\n"
            "Hãy kiểm tra lại đường dẫn root_dir và cấu trúc thư mục."
        )

        # Thu thập tất cả ảnh hợp lệ
        self.samples: list[tuple[Path, int]] = []
        for img_path in sorted(self.color_dir.glob("*.jpg")):
            label = _label_from_name(img_path.name)
            if label != -1:
                self.samples.append((img_path, label))

        assert len(self.samples) > 0, (
            f"Không tìm thấy ảnh nào trong {self.color_dir}\n"
            "File phải có tên kết thúc bằng _real.jpg hoặc _fake.jpg"
        )

        n_real = sum(1 for _, lbl in self.samples if lbl == 1)
        n_fake = len(self.samples) - n_real
        print(f"[CasiaFASDDataset | {split}] Tổng: {len(self.samples)} ảnh "
              f"| Real: {n_real} | Fake: {n_fake}")

        # Transform mặc định nếu không truyền vào
        if transform is not None:
            self.transform = transform
        elif split == "train":
            self.transform = _default_train_transform()
        else:
            self.transform = _default_val_transform()

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        img_path, label = self.samples[idx]
        image = Image.open(img_path).convert("RGB")
        image = self.transform(image)
        return image, torch.tensor(label, dtype=torch.long)

# Augmentation 
def _default_train_transform():
    return transforms.Compose([
        transforms.Resize((128, 128)),           # Resize 
        transforms.RandomResizedCrop(112, scale=(0.8, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.3, contrast=0.3,
                               saturation=0.2, hue=0.05),
        transforms.RandomGrayscale(p=0.05),      
        transforms.ToTensor(),
        # ImageNet mean/std → phù hợp với MobileNetV2 pretrained
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])


def _default_val_transform():
    return transforms.Compose([
        transforms.Resize((112, 112)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

def get_dataloaders(
    root_dir: str,
    batch_size: int = 32,
    num_workers: int = 2,
    train_transform=None,
    val_transform=None,
) -> tuple[DataLoader, DataLoader]:
    """
    Tạo train và val/test DataLoader từ root_dir.

    Returns:
        (train_loader, val_loader)
    """
    train_dataset = CasiaFASDDataset(root_dir, split="train",
                                     transform=train_transform)
    val_dataset   = CasiaFASDDataset(root_dir, split="test",
                                     transform=val_transform)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,       # Tránh batch 1 làm BatchNorm lỗi
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    return train_loader, val_loader

# Test nhanh dataset
if __name__ == "__main__":
    import sys
    root = sys.argv[1] if len(sys.argv) > 1 else "casia-fasd"
    train_loader, val_loader = get_dataloaders(root, batch_size=8, num_workers=0)

    imgs, labels = next(iter(train_loader))
    print(f"Batch shape : {imgs.shape}")   # (8, 3, 112, 112)
    print(f"Labels      : {labels}")
    print(f"Pixel range : [{imgs.min():.2f}, {imgs.max():.2f}]")
