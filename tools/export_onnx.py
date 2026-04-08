import torch
from models.mobilefacenet import MobileFaceNet 
from models.agegendermodel import AgeGenderMobileNetV3

def convert_mobilefacenet_to_onnx(pth_path, onnx_path):
    print("Khởi tạo MobileFaceNet model...")
    model = MobileFaceNet(embedding_size=512)
    
    # Load trọng số
    model.load_state_dict(torch.load(pth_path, map_location=torch.device('cpu'), weights_only=True))
    model.eval() # Chuyển sang chế độ eval trước khi export

    # Tạo một tensor đầu vào giả lập (batch_size=1, channels=3, height=112, width=112)
    dummy_input = torch.randn(1, 3, 112, 112)

    print("Đang xuất MobileFaceNet sang chuẩn ONNX...")
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

def convert_agegendermodel_to_onnx(pth_path, onnx_path):
    print("\nKhởi tạo AgeGenderModel...")
    model = AgeGenderMobileNetV3(pretrained=False)
    
    # Load trọng số
    checkpoint = torch.load(pth_path, map_location=torch.device('cpu'), weights_only=True)
    model.load_state_dict(checkpoint)
    model.eval()  # Chuyển sang chế độ eval trước khi export

    # Tạo một tensor đầu vào giả lập (batch_size=1, channels=3, height=224, width=224)
    dummy_input = torch.randn(1, 3, 224, 224)

    print("Đang xuất AgeGenderModel sang chuẩn ONNX...")
    torch.onnx.export(
        model,
        dummy_input,
        onnx_path,
        export_params=True,
        opset_version=11,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['age', 'gender'],  # 2 outputs: age và gender
        dynamic_axes={
            'input': {0: 'batch_size'},
            'age': {0: 'batch_size'},
            'gender': {0: 'batch_size'}
        }
    )
    print(f"Thành công! File đã được lưu tại: {onnx_path}")

if __name__ == "__main__":
    # Export MobileFaceNet
    MOBILEFACENET_PTH = "saved_models/mobilefacenet.pth"
    MOBILEFACENET_ONNX = "saved_models/mobilefacenet.onnx"
    convert_mobilefacenet_to_onnx(MOBILEFACENET_PTH, MOBILEFACENET_ONNX)
    
    # Export AgeGenderModel
    AGEGENDER_PTH = "saved_models/agegendermodel.pth"
    AGEGENDER_ONNX = "saved_models/agegendermodel.onnx"
    convert_agegendermodel_to_onnx(AGEGENDER_PTH, AGEGENDER_ONNX)