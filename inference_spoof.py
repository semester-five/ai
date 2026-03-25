import cv2
import numpy as np
import onnxruntime as ort
import torch

from models.face_detector import FaceDetector
from models.anti_spoof import AntiSpoofNet

# ==========================================
# 1. CÁC HÀM XỬ LÝ CHO NHẬN DIỆN KHUÔN MẶT
# ==========================================
def load_model(weights_path: str) -> ort.InferenceSession:
    print(f"Đang nạp bộ não Face Recognition ({weights_path})...")
    session = ort.InferenceSession(weights_path, providers=['CPUExecutionProvider'])
    return session

def preprocess_face(face_img: np.ndarray) -> np.ndarray:
    """Tiền xử lý cho MobileFaceNet (ONNX) -> [-1, 1]"""
    face_img = cv2.resize(face_img, (112, 112))
    face_img = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
    face_img = (face_img.astype(np.float32) / 127.5) - 1.0
    face_img = np.transpose(face_img, (2, 0, 1))
    return np.expand_dims(face_img, axis=0)

def get_embedding(session: ort.InferenceSession, input_array: np.ndarray) -> np.ndarray:
    input_name = session.get_inputs()[0].name
    output = session.run(None, {input_name: input_array})
    embedding = output[0].flatten()
    norm = np.linalg.norm(embedding)
    return embedding / norm if norm > 0 else embedding

def cosine_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
    return float(np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2)))

def find_existing_locker(current_vector: np.ndarray, lockers: dict, threshold: float):
    best_id, highest_sim = None, 0.0
    for locker_id, saved_vector in lockers.items():
        if saved_vector is not None:
            sim = cosine_similarity(current_vector, saved_vector)
            if sim > threshold and sim > highest_sim:
                highest_sim = sim
                best_id = locker_id
    return best_id, highest_sim

# ==========================================
# 2. CÁC HÀM XỬ LÝ CHO CHỐNG GIẢ MẠO (LIVENESS)
# ==========================================
def preprocess_for_liveness(face_crop, target_size=(224, 224), device='cpu'):
    """Tiền xử lý cho AntiSpoofNet (PyTorch) -> Chuẩn ImageNet"""
    img_rgb = cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB)
    img_resized = cv2.resize(img_rgb, target_size)
    img_normalized = img_resized / 255.0
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    img_normalized = (img_normalized - mean) / std
    img_final = img_normalized.transpose(2, 0, 1).astype(np.float32)
    return torch.from_numpy(img_final).unsqueeze(0).to(device)

def check_liveness(face_img, liveness_net, device, liveness_threshold):
    """Chạy anti-spoof và trả về (is_real, score, label)."""
    liveness_input = preprocess_for_liveness(face_img, device=device)
    with torch.inference_mode():
        outputs = liveness_net(liveness_input)
        probabilities = torch.softmax(outputs, dim=1)
        score, predicted_idx = torch.max(probabilities, dim=1)
        liveness_score = score.item()
        is_real = (predicted_idx.item() == 1) and (liveness_score >= liveness_threshold)
    return is_real, liveness_score


if __name__ == "__main__":
    # --- CONFIG ---
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    WEIGHTS_PATH = 'saved_models/mobilefacenet_int8.onnx'
    LIVENESS_PATH = 'saved_models/anti_spoof.pth'

    RECOGNITION_THRESHOLD = 0.60
    LIVENESS_THRESHOLD = 0.90

    # --- NẠP MODEL ---
    face_detector = FaceDetector()
    session = load_model(WEIGHTS_PATH)

    print(f"Đang nạp bộ não Anti-Spoofing ({LIVENESS_PATH}) trên {DEVICE}...")
    liveness_net = AntiSpoofNet(pretrained=False).to(DEVICE)
    liveness_net.load_state_dict(torch.load(LIVENESS_PATH, map_location=DEVICE))
    liveness_net.eval()
    print("-> Model Liveness loaded.")

    # --- DATABASE TỦ ĐỒ (RAM) ---
    lockers = {1: None, 2: None}

    # --- START WEBCAM ---
    cap = cv2.VideoCapture(0)
    print("\n" + "="*40)
    print(" KIOSK TỦ ĐỒ THÔNG MINH (TÍCH HỢP ANTI-SPOOFING)")
    print("="*40)
    print("Nhấn 'I' để Cất đồ (Check-in)")
    print("Nhấn 'O' để Lấy đồ (Check-out)")
    print("Nhấn 'Q' để Thoát")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        display_frame = frame.copy()
        ih, iw, _ = frame.shape

        detection = face_detector.detect(frame)
        face_img = None

        if detection is not None:
            x, y, w, h = detection
            x1, y1 = max(0, x - 20), max(0, y - 20)
            x2, y2 = min(iw, x + w + 20), min(ih, y + h + 20)

            if w > 10 and h > 10:
                face_img = frame[y1:y2, x1:x2]
                # Khung xanh cố định, không đổi màu theo liveness
                cv2.rectangle(display_frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        cv2.putText(display_frame, "I: Cat do | O: Lay do | Q: Thoat",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        cv2.imshow("Kiosk FaceID & Anti-Spoof", display_frame)

        key = cv2.waitKey(1) & 0xFF

        # --- LOGIC NHẤN PHÍM ---
        if key == ord('q'):
            print("Đang tắt hệ thống...")
            break

        elif key == ord('i') or key == ord('o'):
            action = "CẤT ĐỒ" if key == ord('i') else "LẤY ĐỒ"
            print(f"\n[{action} REQUEST]")

            if face_img is None or face_img.size == 0:
                print("-> ❌ Không tìm thấy khuôn mặt. Vui lòng nhìn thẳng vào camera.")
                continue

            # BƯỚC 1: Kiểm tra liveness tại thời điểm nhấn phím
            is_real, liveness_score = check_liveness(
                face_img, liveness_net, DEVICE, LIVENESS_THRESHOLD
            )

            if not is_real:
                print(f"-> 🚨 CẢNH BÁO: Phát hiện khuôn mặt GIẢ! (Score: {liveness_score:.2f})")
                print("-> Yêu cầu bị từ chối.")
                continue

            print(f"-> ✔ Khuôn mặt THẬT (Score: {liveness_score:.2f}). Tiến hành xác thực...")

            # BƯỚC 2: Người thật -> trích xuất đặc trưng & xử lý tủ
            input_array = preprocess_face(face_img)
            current_vector = get_embedding(session, input_array)

            if key == ord('i'):  # CHECK-IN
                existing_locker, sim = find_existing_locker(
                    current_vector, lockers, RECOGNITION_THRESHOLD
                )
                if existing_locker is not None:
                    print(f"-> ⚠️  Bạn đang có đồ ở tủ #{existing_locker} "
                          f"(Độ khớp: {sim:.2f}). Vui lòng lấy đồ ra trước.")
                else:
                    empty_locker = next(
                        (lid for lid, vec in lockers.items() if vec is None), None
                    )
                    if empty_locker is None:
                        print("-> ❌ Rất tiếc, tất cả các tủ đều đã đầy.")
                    else:
                        lockers[empty_locker] = current_vector
                        print(f"-> ✅ Cất đồ thành công! Tủ #{empty_locker} đang mở.")

            elif key == ord('o'):  # CHECK-OUT
                matched_locker, highest_sim = find_existing_locker(
                    current_vector, lockers, RECOGNITION_THRESHOLD
                )
                if matched_locker is not None:
                    print(f"-> ✅ Xác thực thành công (Độ khớp: {highest_sim:.2f}). "
                          f"Tủ #{matched_locker} đang mở.")
                    lockers[matched_locker] = None
                else:
                    print("-> ❌ Khuôn mặt không khớp với bất kỳ tủ nào đang sử dụng.")

    cap.release()
    cv2.destroyAllWindows()