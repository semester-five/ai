import cv2
import numpy as np
import onnxruntime as ort
from torchvision import transforms

from models.face_detector import FaceDetector


def load_model(weights_path: str) -> ort.InferenceSession:
    print("Đang nạp bộ não AI...")
    session = ort.InferenceSession(
        weights_path,
        providers=['CUDAExecutionProvider', 'CPUExecutionProvider']
    )
    input_shape = session.get_inputs()[0].shape
    print(f"-> Model loaded. Input shape: {input_shape}")
    return session


def preprocess_face(face_img: np.ndarray) -> np.ndarray:
    """Pre-process a cropped face image into a model-ready numpy array."""
    face_img = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((112, 112)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    # ONNX Runtime expects a plain numpy array, not a torch.Tensor
    tensor = transform(face_img).unsqueeze(0)   # (1, 3, 112, 112)
    return tensor.numpy().astype(np.float32)    # → numpy float32


def get_embedding(session: ort.InferenceSession, input_array: np.ndarray) -> np.ndarray:
    """Run a forward pass and return the L2-normalised embedding vector."""
    input_name = session.get_inputs()[0].name
    output = session.run(None, {input_name: input_array})
    embedding = output[0].flatten()
    # Normalise just in case the exported model omits the final L2 norm
    norm = np.linalg.norm(embedding)
    return embedding / norm if norm > 0 else embedding


def cosine_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
    return float(np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2)))


def find_existing_locker(
    current_vector: np.ndarray,
    lockers: dict,
    threshold: float
) -> tuple[int | None, float]:
    """
    Check whether the current face is already registered to a locker.
    Returns (locker_id, similarity) if found, otherwise (None, 0.0).
    """
    best_id, highest_sim = None, 0.0
    for locker_id, saved_vector in lockers.items():
        if saved_vector is not None:
            sim = cosine_similarity(current_vector, saved_vector)
            if sim > threshold and sim > highest_sim:
                highest_sim = sim
                best_id = locker_id
    return best_id, highest_sim

if __name__ == "__main__":
    # 1. Config
    WEIGHTS_PATH = 'saved_models/mobilefacenet.onnx'
    THRESHOLD = 0.60

    session = load_model(WEIGHTS_PATH)
    face_detector = FaceDetector()

    # 2. Locker database 
    lockers = {1: None, 2: None}

    # 3. Start webcam
    cap = cv2.VideoCapture(0)
    print("\n--- LOCKER KIOSK SYSTEM READY ---")
    print("Press 'I' to Check-in  (store items)")
    print("Press 'O' to Check-out (retrieve items)")
    print("Press 'Q' to Quit")

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
            face_img = frame[y1:y2, x1:x2]
            cv2.rectangle(display_frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        cv2.putText(display_frame, "I: CHECK-IN | O: CHECK-OUT | Q: QUIT",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        cv2.imshow("Kiosk FaceID", display_frame)
        key = cv2.waitKey(1) & 0xFF

        # --- KEY LOGIC ---
        if key == ord('q'):
            print("Shutting down...")
            break

        elif key == ord('i'):
            print("\n[CHECK-IN REQUEST]")
            if face_img is None or face_img.size == 0:
                print("-> No face detected. Please look directly at the camera.")
                continue

            input_array = preprocess_face(face_img)
            current_vector = get_embedding(session, input_array)

            # Block duplicate check-in
            existing_locker, sim = find_existing_locker(current_vector, lockers, THRESHOLD)
            if existing_locker is not None:
                print(f"-> ⚠️  WARNING: You already have items in locker #{existing_locker} "
                      f"(similarity: {sim:.2f}). Please check out first.")
                continue

            # Find an empty locker
            empty_locker = next(
                (lid for lid, vec in lockers.items() if vec is None), None
            )
            if empty_locker is None:
                print("-> Sorry, all lockers are currently occupied.")
            else:
                lockers[empty_locker] = current_vector
                print(f"-> Check-in successful! Locker #{empty_locker} is now open.")

        elif key == ord('o'):
            print("\n[CHECK-OUT REQUEST]")
            if face_img is None or face_img.size == 0:
                print("-> No face detected. Please look directly at the camera.")
                continue

            input_array = preprocess_face(face_img)
            current_vector = get_embedding(session, input_array)

            matched_locker, highest_sim = find_existing_locker(current_vector, lockers, THRESHOLD)
            if matched_locker is not None:
                print(f"-> Identity verified (similarity: {highest_sim:.2f}). "
                      f"Locker #{matched_locker} is now open.")
                lockers[matched_locker] = None
            else:
                print("-> Face does not match any active locker.")

    cap.release()
    cv2.destroyAllWindows()