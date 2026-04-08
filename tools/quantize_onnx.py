from onnxruntime.quantization import quantize_dynamic, QuantType

def quantize_onnx_model(onnx_input_path, onnx_output_path, model_name="Model"):
    print(f"Đang đọc file gốc: {onnx_input_path}")
    
    # Thực hiện ép toàn bộ trọng số mạng về số nguyên 8-bit (QUInt8)
    quantize_dynamic(
        model_input=onnx_input_path,
        model_output=onnx_output_path,
        weight_type=QuantType.QUInt8 
    )
    
    print(f"Hoàn tất! File {model_name} ONNX đã được lượng tử hóa lưu tại: {onnx_output_path}")

if __name__ == "__main__":
    # Quantize MobileFaceNet
    print("=" * 50)
    MOBILEFACENET_IN = "saved_models/mobilefacenet.onnx"
    MOBILEFACENET_OUT = "saved_models/mobilefacenet_int8.onnx"
    quantize_onnx_model(MOBILEFACENET_IN, MOBILEFACENET_OUT, "MobileFaceNet")
    
    # Quantize AgeGenderModel
    print("=" * 50)
    AGEGENDER_IN = "saved_models/agegendermodel.onnx"
    AGEGENDER_OUT = "saved_models/agegendermodel_int8.onnx"
    quantize_onnx_model(AGEGENDER_IN, AGEGENDER_OUT, "AgeGenderModel")
    
    print("=" * 50)
    print("✅ Tất cả model đã được lượng tử hóa thành công!")