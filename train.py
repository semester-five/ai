import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

# Import các module bạn đã định nghĩa
from utils.dataset import get_dataloaders
from models.anti_spoof import AntiSpoofNet, freeze_backbone, unfreeze_last_n_layers

def train_one_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    # Hiển thị thanh tiến trình
    pbar = tqdm(dataloader, desc="Training", leave=False)
    for images, labels in pbar:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

        pbar.set_postfix({"loss": f"{loss.item():.4f}"})

    epoch_loss = running_loss / total
    epoch_acc = 100. * correct / total
    return epoch_loss, epoch_acc

def validate_one_epoch(model, dataloader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.inference_mode():
        pbar = tqdm(dataloader, desc="Validating", leave=False)
        for images, labels in pbar:
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * images.size(0)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    epoch_loss = running_loss / total
    epoch_acc = 100. * correct / total
    return epoch_loss, epoch_acc

def main():
    # Khởi tạo bộ đọc tham số từ dòng lệnh
    parser = argparse.ArgumentParser(description="Huấn luyện mô hình Anti-Spoofing")
    parser.add_argument("--data_root", type=str, default="data/casia-fasd", help="Đường dẫn đến thư mục chứa dữ liệu")
    parser.add_argument("--batch_size", type=int, default=32, help="Kích thước batch (giảm nếu hết VRAM)")
    parser.add_argument("--num_workers", type=int, default=2, help="Số luồng đọc dữ liệu")
    parser.add_argument("--epochs_p1", type=int, default=5, help="Số epoch cho Phase 1 (Đóng băng backbone)")
    parser.add_argument("--epochs_p2", type=int, default=15, help="Số epoch cho Phase 2 (Fine-tune)")
    parser.add_argument("--lr_p1", type=float, default=1e-3, help="Learning rate cho Phase 1")
    parser.add_argument("--lr_p2", type=float, default=1e-4, help="Learning rate cho Phase 2 (Nên để nhỏ)")
    parser.add_argument("--save_dir", type=str, default="saved_models", help="Thư mục lưu trọng số model")
    args = parser.parse_args()

    # Tạo thư mục lưu model nếu chưa có
    os.makedirs(args.save_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[*] Đang sử dụng thiết bị tính toán: {device}")

    # 1. Chuẩn bị dữ liệu
    print("[*] Đang nạp dữ liệu...")
    train_loader, val_loader = get_dataloaders(
        root_dir=args.data_root, 
        batch_size=args.batch_size, 
        num_workers=args.num_workers
    )

    # 2. Khởi tạo mô hình và hàm Loss
    print("[*] Đang khởi tạo mô hình...")
    model = AntiSpoofNet(pretrained=True).to(device)
    criterion = nn.CrossEntropyLoss()

    best_val_loss = float('inf')
    save_path = os.path.join(args.save_dir, "anti_spoof.pth")
    global_epoch = 1

    # ==========================================
    # PHASE 1: Train Classifier Only
    # ==========================================
    print("\n" + "="*50)
    print(" PHASE 1: ĐÓNG BĂNG BACKBONE (Train Classifier)")
    print("="*50)
    freeze_backbone(model)
    
    # Chỉ truyền các tham số requires_grad=True vào Optimizer
    optimizer_p1 = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr_p1)

    for epoch in range(1, args.epochs_p1 + 1):
        print(f"\n[Global Epoch {global_epoch}] Phase 1 - Epoch {epoch}/{args.epochs_p1}")
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer_p1, device)
        val_loss, val_acc = validate_one_epoch(model, val_loader, criterion, device)
        
        print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
        print(f"Val Loss:   {val_loss:.4f} | Val Acc:   {val_acc:.2f}%")
        
        # Lưu model nếu Validation Loss tốt hơn
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), save_path)
            print(f"-> Đã lưu model tốt nhất vào: {save_path}")
            
        global_epoch += 1

    # ==========================================
    # PHASE 2: Fine-tune Last N Layers
    # ==========================================
    print("\n" + "="*50)
    print(" PHASE 2: MỞ BĂNG 5 BLOCK CUỐI (Fine-tuning)")
    print("="*50)
    unfreeze_last_n_layers(model, n=5)
    
    # Khởi tạo lại Optimizer để nhận diện các tham số vừa được mở băng, dùng LR nhỏ hơn
    optimizer_p2 = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr_p2)

    for epoch in range(1, args.epochs_p2 + 1):
        print(f"\n[Global Epoch {global_epoch}] Phase 2 - Epoch {epoch}/{args.epochs_p2}")
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer_p2, device)
        val_loss, val_acc = validate_one_epoch(model, val_loader, criterion, device)
        
        print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
        print(f"Val Loss:   {val_loss:.4f} | Val Acc:   {val_acc:.2f}%")
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), save_path)
            print(f"-> Đã lưu model tốt nhất vào: {save_path}")
            
        global_epoch += 1

    print(f"\n[*] Hoàn thành huấn luyện! Best Val Loss: {best_val_loss:.4f}")

if __name__ == "__main__":
    main()