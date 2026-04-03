import cv2
import tkinter as tk
from tkinter import font as tkfont
from PIL import Image, ImageTk
import numpy as np
import onnxruntime as ort
import threading

# Import lại các hàm và module từ code cũ của bạn
from inference_spoof import load_model, preprocess_face, get_embedding, check_liveness, find_existing_locker
from models.face_detector import FaceDetector

class SmartLockerGUI:
    def __init__(self, window, window_title):
        self.window = window
        self.window.title(window_title)
        self.window.geometry("1000x600")
        self.window.configure(bg="#2C3E50") # Màu nền tối hiện đại

        # --- CẤU HÌNH AI & DATABASE ---
        self.WEIGHTS_PATH = 'saved_models/mobilefacenet_int8.onnx'
        self.LIVENESS_PATH = 'saved_models/anti_spoof.onnx'
        self.RECOGNITION_THRESHOLD = 0.60
        self.LIVENESS_THRESHOLD = 0.90
        self.lockers = {1: None, 2: None, 3: None, 4: None} # Mở rộng thêm tủ
        
        self.face_detector = None
        self.session = None
        self.liveness_session = None
        self.current_face_img = None # Lưu khuôn mặt hiện tại để quét

        # --- SETUP GIAO DIỆN ---
        self.setup_ui()
        
        # Load Model trong luồng riêng để không bị đơ giao diện
        threading.Thread(target=self.init_models, daemon=True).start()

        # --- KẾT NỐI CAMERA RASPBERRY PI ---
        self.video_source = "http://100.118.31.76:5000/video"
        self.vid = cv2.VideoCapture(self.video_source)
        
        if not self.vid.isOpened():
            self.show_message("LỖI: Không thể kết nối đến Camera Raspberry Pi!", "red")
        else:
            self.update_frame()

    def setup_ui(self):
        # Font chữ
        title_font = tkfont.Font(family="Helvetica", size=18, weight="bold")
        btn_font = tkfont.Font(family="Helvetica", size=14, weight="bold")
        msg_font = tkfont.Font(family="Helvetica", size=12)

        # Khung chứa Camera (Trái)
        self.video_frame = tk.Frame(self.window, bg="#34495E", width=640, height=480)
        self.video_frame.pack(side=tk.LEFT, padx=20, pady=20)
        
        self.canvas = tk.Canvas(self.video_frame, width=640, height=480, bg="black", highlightthickness=0)
        self.canvas.pack()

        # Khung chứa Bảng điều khiển (Phải)
        self.control_frame = tk.Frame(self.window, bg="#2C3E50")
        self.control_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=20, pady=20)

        lbl_title = tk.Label(self.control_frame, text="KIOSK TỦ ĐỒ THÔNG MINH", font=title_font, fg="#ECF0F1", bg="#2C3E50")
        lbl_title.pack(pady=(0, 30))

        # Nút Bấm
        self.btn_checkin = tk.Button(self.control_frame, text="📥 QUÉT MẶT - CẤT ĐỒ", font=btn_font, 
                                     bg="#27AE60", fg="white", activebackground="#2ECC71", 
                                     command=lambda: self.process_scan("check_in"))
        self.btn_checkin.pack(fill=tk.X, pady=10, ipady=10)

        self.btn_checkout = tk.Button(self.control_frame, text="📤 QUÉT MẶT - LẤY ĐỒ", font=btn_font, 
                                      bg="#2980B9", fg="white", activebackground="#3498DB", 
                                      command=lambda: self.process_scan("check_out"))
        self.btn_checkout.pack(fill=tk.X, pady=10, ipady=10)

        # Khung Thông Báo
        self.msg_frame = tk.LabelFrame(self.control_frame, text="Trạng thái hệ thống", font=msg_font, fg="#BDC3C7", bg="#2C3E50")
        self.msg_frame.pack(fill=tk.BOTH, expand=True, pady=30)

        self.lbl_message = tk.Label(self.msg_frame, text="Đang khởi động hệ thống...", font=msg_font, 
                                    fg="#F1C40F", bg="#2C3E50", wraplength=250, justify="center")
        self.lbl_message.pack(expand=True)

    def init_models(self):
        self.show_message("Đang nạp AI Models...", "#F1C40F")
        try:
            self.face_detector = FaceDetector()
            self.session = load_model(self.WEIGHTS_PATH)
            self.liveness_session = ort.InferenceSession(self.LIVENESS_PATH, providers=['CPUExecutionProvider'])
            self.show_message("Hệ thống sẵn sàng!\nVui lòng nhìn vào camera.", "#2ECC71")
        except Exception as e:
            self.show_message(f"Lỗi nạp model: {str(e)}", "#E74C3C")

    def show_message(self, text, color):
        self.window.after(0, lambda: self.lbl_message.config(text=text, fg=color))

    def update_frame(self):
        ret, frame = self.vid.read()
        if ret:
            # Lật ảnh để giống gương
            frame = cv2.flip(frame, 1)
            display_frame = frame.copy()
            ih, iw, _ = frame.shape
            
            self.current_face_img = None # Reset mỗi frame

            if self.face_detector is not None:
                detection = self.face_detector.detect(frame)
                if detection is not None:
                    x, y, w, h = detection
                    pad_x, pad_y = int(w * 0.7), int(h * 0.7)
                    x1, y1 = max(0, x - pad_x), max(0, y - pad_y)
                    x2, y2 = min(iw, x + w + pad_x), min(ih, y + h + pad_y)

                    if w > 10 and h > 10:
                        self.current_face_img = frame[y1:y2, x1:x2]
                        cv2.rectangle(display_frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

            # Chuyển đổi BGR sang RGB để hiển thị trên Tkinter
            cv2image = cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(cv2image)
            imgtk = ImageTk.PhotoImage(image=img)
            
            self.canvas.imgtk = imgtk
            self.canvas.create_image(0, 0, anchor=tk.NW, image=imgtk)

        # Lặp lại việc vẽ frame sau mỗi 15ms (~60fps)
        self.window.after(15, self.update_frame)

    def process_scan(self, action):
        if self.session is None or self.liveness_session is None:
            self.show_message("Hệ thống AI chưa sẵn sàng!", "#E74C3C")
            return

        if self.current_face_img is None or self.current_face_img.size == 0:
            self.show_message("❌ Không tìm thấy khuôn mặt.\nVui lòng nhìn thẳng camera.", "#E74C3C")
            return

        # 1. Check Liveness
        is_real, liveness_score = check_liveness(self.current_face_img, self.liveness_session, self.LIVENESS_THRESHOLD)
        if not is_real:
            self.show_message(f"🚨 PHÁT HIỆN GIẢ MẠO!\nYêu cầu bị từ chối.\n(Score: {liveness_score:.2f})", "#E74C3C")
            return

        # 2. Extract Embedding
        input_array = preprocess_face(self.current_face_img)
        current_vector = get_embedding(self.session, input_array)

        # 3. Xử lý Logic
        if action == "check_in":
            existing_locker, sim = find_existing_locker(current_vector, self.lockers, self.RECOGNITION_THRESHOLD)
            if existing_locker is not None:
                self.show_message(f"⚠️ Bạn đang có đồ ở tủ #{existing_locker}\nVui lòng lấy đồ ra trước.", "#F39C12")
            else:
                empty_locker = next((lid for lid, vec in self.lockers.items() if vec is None), None)
                if empty_locker is None:
                    self.show_message("❌ Rất tiếc, tất cả các tủ đều đã đầy.", "#E74C3C")
                else:
                    self.lockers[empty_locker] = current_vector
                    self.show_message(f"✅ CẤT ĐỒ THÀNH CÔNG!\n\nTủ #{empty_locker} đang mở.", "#2ECC71")

        elif action == "check_out":
            matched_locker, highest_sim = find_existing_locker(current_vector, self.lockers, self.RECOGNITION_THRESHOLD)
            if matched_locker is not None:
                self.lockers[matched_locker] = None # Trống tủ
                self.show_message(f"✅ XÁC THỰC THÀNH CÔNG!\n\nTủ #{matched_locker} đang mở.\n(Độ khớp: {highest_sim:.2f})", "#3498DB")
            else:
                self.show_message("❌ Khuôn mặt không khớp\nvới bất kỳ tủ nào đang sử dụng.", "#E74C3C")

    def __del__(self):
        if hasattr(self, 'vid') and self.vid.isOpened():
            self.vid.release()

if __name__ == "__main__":
    root = tk.Tk()
    app = SmartLockerGUI(root, "Hệ Thống Tủ Đồ Thông Minh")
    root.mainloop()