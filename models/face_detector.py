import cv2

class FaceDetector:
    """
    Sử dụng YuNet tích hợp sẵn trong OpenCV.
    Siêu nhẹ (1.8MB), tốc độ cực nhanh trên CPU 
    """
    def __init__(self, model_path="saved_models/face_detection_yunet_2023mar.onnx", conf_threshold=0.6):
        # Khởi tạo FaceDetectorYN của OpenCV
        self.detector = cv2.FaceDetectorYN.create(
            model=model_path,
            config="",
            input_size=(300, 300), # Kích thước khung
            score_threshold=conf_threshold,
            nms_threshold=0.3,
            top_k=5000
        )

    def detect(self, frame):
        """Trả về (x, y, w, h) của khuôn mặt rõ nhất."""
        height, width, _ = frame.shape
     
        self.detector.setInputSize((width, height))
        
        # Nhận diện
        _, faces = self.detector.detect(frame)
        
        # Nếu không tìm thấy ai
        if faces is None:
            return None
            
        # faces là một mảng numpy, mỗi hàng là 1 khuôn mặt.
        # Lấy khuôn mặt có độ tin cậy (score - ở vị trí index 14) cao nhất
        best_face = max(faces, key=lambda f: f[14])
        
        x, y, w, h = int(best_face[0]), int(best_face[1]), int(best_face[2]), int(best_face[3])
        
        # Chống tràn viền 
        x = max(0, x)
        y = max(0, y)
        
        return (x, y, w, h)