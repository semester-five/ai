import torch
from models.mobilefacenet import MobileFaceNet 

def convert_pth_to_onnx(pth_path, onnx_path):
    print("Khởi tạo model...")
    model = MobileFaceNet(embedding_size=512)
    
    # Load trọng số
    model.load_state_dict(torch.load(pth_path, map_location=torch.device('cpu'), weights_only=True))
    model.eval() # Chuyển sang chế độ eval trước khi export

    # Tạo một tensor đầu vào giả lập (batch_size=1, channels=3, height=112, width=112)
    dummy_input = torch.randn(1, 3, 112, 112)

    print("Đang xuất file sang chuẩn ONNX...")
    torch.onnx.export(
        model, 
        dummy_input, 
        onnx_path,
        export_params=True,
        opset_version=11,          # Opset 11 là phổ biến và tương thích tốt nhất
        do_constant_folding=True,  # Tối ưu hóa model
        input_names=['input'],     # Đặt tên cho node đầu vào
        output_names=['output'],   # Đặt tên cho node đầu ra
        dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}} # Hỗ trợ batch_size động
    )
    print(f"Thành công! File đã được lưu tại: {onnx_path}")

if __name__ == "__main__":
    PTH_FILE = "saved_models/mobilefacenet.pth"
    ONNX_FILE = "saved_models/mobilefacenet.onnx"
    convert_pth_to_onnx(PTH_FILE, ONNX_FILE)