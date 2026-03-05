import cv2
import torch
import numpy as np
from torchvision import transforms

# Import model từ thư mục 
from models.mobilefacenet import MobileFaceNet
from models.face_detector import FaceDetector

def load_model(weights_path, device='cpu'):
    print("Đang nạp bộ não AI...")
    model = MobileFaceNet(embedding_size=128).to(device)
    model.load_state_dict(torch.load(weights_path, map_location=device, weights_only=True))
    model.eval()
    return model

def preprocess_face(face_img):
    """Tiền xử lý khuôn mặt đã được cắt từ Webcam."""
    face_img = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((112, 112)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    return transform(face_img).unsqueeze(0)

def get_embedding(model, img_tensor, device='cpu'):
    """Trích xuất vector 128 chiều."""
    img_tensor = img_tensor.to(device)
    with torch.no_grad():
        feature = model(img_tensor)
    return feature.cpu().numpy().flatten()

def cosine_similarity(vec1, vec2):
    return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))

if __name__ == "__main__":
    WEIGHTS_PATH = 'saved_models/mobilefacenet.pth'
    THRESHOLD = 0.45
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = load_model(WEIGHTS_PATH, device)
    face_detector = FaceDetector()
    lockers = {1: None, 2: None}

    cap = cv2.VideoCapture(0)
    print("\n--- HỆ THỐNG KIOSK TỦ ĐỒ SẴN SÀNG ---")
    print("Nhấn 'I' để Check-in (Gửi đồ)")
    print("Nhấn 'O' để Check-out (Lấy đồ)")
    print("Nhấn 'Q' để Thoát")

    while True:
        ret, frame = cap.read()
        if not ret: break
        frame = cv2.flip(frame, 1)
        display_frame = frame.copy()

        # Tạm thời chỉ hiển thị Text và khung hình
        cv2.putText(display_frame, "I: GUi DO | O: LAY DO | Q: THOAT", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        cv2.imshow("Kiosk FaceID", display_frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            print("Đang tắt hệ thống...")
            break

    cap.release()
    cv2.destroyAllWindows()