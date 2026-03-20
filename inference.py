import cv2
import torch
import numpy as np
from torchvision import transforms

# Import model từ thư mục 
from models.mobilefacenet import MobileFaceNet
from models.face_detector import FaceDetector

def load_model(weights_path, device='cpu'):
    print("Đang nạp bộ não AI...")
    model = MobileFaceNet(embedding_size=512).to(device)
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
    """Trích xuất vector 512 chiều."""
    img_tensor = img_tensor.to(device)
    with torch.no_grad():
        feature = model(img_tensor)
    return feature.cpu().numpy().flatten()

def cosine_similarity(vec1, vec2):
    return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))

def find_existing_locker(current_vector, lockers, threshold):
    """
    Kiểm tra xem khuôn mặt hiện tại đã đăng ký tủ nào chưa.
    Trả về (locker_id, similarity) nếu tìm thấy, hoặc (None, 0) nếu chưa.
    """
    best_match_id = None
    highest_sim = 0

    for locker_id, saved_vector in lockers.items():
        if saved_vector is not None:
            sim = cosine_similarity(current_vector, saved_vector)
            if sim > threshold and sim > highest_sim:
                highest_sim = sim
                best_match_id = locker_id

    return best_match_id, highest_sim

# Main
if __name__ == "__main__":
    # 1. Cấu hình
    WEIGHTS_PATH = 'saved_models/mobilefacenet.pth'
    THRESHOLD = 0.60
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = load_model(WEIGHTS_PATH, device)

    # Khởi tạo Face Detector 
    face_detector = FaceDetector()

    # 2. Giả lập cơ sở dữ liệu tủ đồ 
    lockers = {1: None, 2: None}

    # 3. Khởi động Webcam
    cap = cv2.VideoCapture(0)
    print("\n--- HỆ THỐNG KIOSK TỦ ĐỒ SẴN SÀNG ---")
    print("Nhấn 'I' để Check-in (Gửi đồ)")
    print("Nhấn 'O' để Check-out (Lấy đồ)")
    print("Nhấn 'Q' để Thoát")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Lật khung hình cho giống gương
        frame = cv2.flip(frame, 1)
        display_frame = frame.copy()

        # Tìm khuôn mặt
        ih, iw, _ = frame.shape
        detection = face_detector.detect(frame)
        face_img = None

        if detection is not None:
            x, y, w, h = detection
            # Cắt khuôn mặt (mở rộng thêm viền)
            x1 = max(0, x - 20)
            y1 = max(0, y - 20)
            x2 = min(iw, x + w + 20)
            y2 = min(ih, y + h + 20)
            face_img = frame[y1:y2, x1:x2]
            cv2.rectangle(display_frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # Hướng dẫn hiển thị trên màn hình
        cv2.putText(display_frame, "I: GUi DO | O: LAY DO | Q: THOAT", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

        cv2.imshow("Kiosk FaceID", display_frame)
        key = cv2.waitKey(1) & 0xFF

        # --- LOGIC NHẤN PHÍM ---
        if key == ord('q'):
            print("Đang tắt hệ thống...")
            break

        elif key == ord('i'):
            print("\n[YÊU CẦU GỬI ĐỒ]")
            if face_img is None or face_img.size == 0:
                print("-> Không tìm thấy khuôn mặt! Vui lòng nhìn thẳng vào camera.")
                continue

            tensor = preprocess_face(face_img)
            current_vector = get_embedding(model, tensor, device)

            # ✅ KIỂM TRA: Khuôn mặt này đã đăng ký tủ nào chưa?
            existing_locker, sim = find_existing_locker(current_vector, lockers, THRESHOLD)
            if existing_locker is not None:
                print(f"-> ⚠️  CẢNH BÁO: Bạn đã gửi đồ ở tủ số {existing_locker} rồi! (Độ tin cậy: {sim:.2f})")
                print("   Vui lòng lấy đồ ra trước khi gửi lại.")
                continue

            # Tìm tủ trống
            empty_locker = None
            for locker_id, saved_vector in lockers.items():
                if saved_vector is None:
                    empty_locker = locker_id
                    break

            if empty_locker is None:
                print("-> Xin lỗi, hiện tại đã hết tủ trống!")
            else:
                lockers[empty_locker] = current_vector  # Tái dùng vector đã tính, không cần tính lại
                print(f"-> Gửi đồ thành công! Tủ số {empty_locker} đã mở. (Đã lưu dữ liệu khuôn mặt)")

        elif key == ord('o'):
            print("\n[YÊU CẦU LẤY ĐỒ]")
            if face_img is None or face_img.size == 0:
                print("-> Không tìm thấy khuôn mặt! Vui lòng nhìn thẳng vào camera.")
                continue

            tensor = preprocess_face(face_img)
            current_vector = get_embedding(model, tensor, device)

            matched_locker, highest_sim = find_existing_locker(current_vector, lockers, THRESHOLD)

            if matched_locker is not None:
                print(f"-> Xác thực thành công (Độ tin cậy: {highest_sim:.2f}). Mở tủ số {matched_locker}!")
                lockers[matched_locker] = None
            else:
                print("-> Khuôn mặt không khớp với bất kỳ tủ nào đang gửi đồ!")

    cap.release()
    cv2.destroyAllWindows()