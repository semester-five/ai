from onnxruntime.quantization import quantize_dynamic, QuantType

def quantize_onnx_model(onnx_input_path, onnx_output_path):
    print(f"Đang đọc file gốc: {onnx_input_path}")
    
    # Thực hiện ép toàn bộ trọng số mạng về số nguyên 8-bit (QUInt8)
    quantize_dynamic(
        model_input=onnx_input_path,
        model_output=onnx_output_path,
        weight_type=QuantType.QUInt8 
    )
    
    print(f"Hoàn tất! File ONNX đã được lượng tử hóa lưu tại: {onnx_output_path}")

if __name__ == "__main__":
    IN_PATH = "saved_models/mobilefacenet.onnx"
    OUT_PATH = "saved_models/mobilefacenet_int8.onnx"
    quantize_onnx_model(IN_PATH, OUT_PATH)