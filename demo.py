import cv2
import tkinter as tk
from tkinter import font as tkfont
from PIL import Image, ImageTk
import numpy as np
import onnxruntime as ort
import threading
import queue

# Import các hàm từ inference.py
from inference import (
    load_model, preprocess_face, get_embedding, find_existing_locker,
    load_age_gender_model, predict_age_gender
)
from models.face_detector import FaceDetector

class SmartLockerGUI:
    def __init__(self, window, window_title):
        self.window = window
        self.window.title(window_title)
        self.window.geometry("1600x900")
        self.window.configure(bg="#1A1A1A")
        
        # Hàng đợi tin nhắn
        self.msg_queue = queue.Queue()
        self.user_info_queue = queue.Queue()

        # --- CẤU HÌNH AI & DATABASE ---
        self.WEIGHTS_PATH = 'saved_models/mobilefacenet_int8.onnx'
        self.AGE_GENDER_WEIGHTS = 'saved_models/agegendermodel_int8.onnx'
        self.RECOGNITION_THRESHOLD = 0.60
        self.lockers = {1: None, 2: None, 3: None, 4: None}
        
        self.face_detector = None
        self.session = None
        self.age_gender_session = None
        self.current_face_img = None

        self.process_queue()

        # --- SETUP GIAO DIỆN ---
        self.setup_ui()
        
        # Load Models
        threading.Thread(target=self.init_models, daemon=True).start()

        # --- KẾT NỐI CAMERA ---
        # Dùng webcam cục bộ (0) hoặc camera Raspberry Pi (URL)
        self.video_source = 0
        self.vid = cv2.VideoCapture(self.video_source)
        
        if not self.vid.isOpened():
            self.show_message("LỖI: Không thể kết nối camera!", "error")
        else:
            self.update_frame()

    def setup_ui(self):
        # Fonts
        title_font = tkfont.Font(family="Helvetica", size=20, weight="bold")
        header_font = tkfont.Font(family="Helvetica", size=12, weight="bold")
        btn_font = tkfont.Font(family="Helvetica", size=12, weight="bold")
        msg_font = tkfont.Font(family="Helvetica", size=11)
        info_font = tkfont.Font(family="Helvetica", size=10)

        # ===== KHUNG TRÁI: CAMERA =====
        left_frame = tk.Frame(self.window, bg="#2A2A2A")
        left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=10, pady=10)

        lbl_camera = tk.Label(left_frame, text="📹 CAMERA", font=header_font, fg="#00D4FF", bg="#2A2A2A")
        lbl_camera.pack(pady=(0, 10))

        # Canvas wrapper để center
        camera_wrapper = tk.Frame(left_frame, bg="#2A2A2A")
        camera_wrapper.pack(expand=True, fill=tk.BOTH)

        self.canvas = tk.Canvas(camera_wrapper, width=800, height=600, bg="#000000", highlightthickness=3, highlightbackground="#00D4FF")
        self.canvas.pack(expand=True, fill=tk.BOTH)

        # ===== KHUNG PHẢI: ĐIỀU KHIỂN =====
        right_frame = tk.Frame(self.window, bg="#1A1A1A", width=350)
        right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, padx=10, pady=10)
        right_frame.pack_propagate(False)

        # Tiêu đề
        lbl_title = tk.Label(right_frame, text="🔐 KIOSK TỦ ĐỒ", font=title_font, fg="#00D4FF", bg="#1A1A1A")
        lbl_title.pack(pady=(0, 20))

        # ===== CÁC NÚT BẤM =====
        btn_frame = tk.Frame(right_frame, bg="#1A1A1A")
        btn_frame.pack(fill=tk.X, pady=(0, 15))

        self.btn_checkin = tk.Button(
            btn_frame,
            text="📥 CẤT ĐỒ",
            font=btn_font,
            bg="#27AE60",
            fg="white",
            activebackground="#2ECC71",
            activeforeground="white",
            relief=tk.RAISED,
            bd=2,
            padx=10,
            pady=10,
            command=lambda: self.process_scan("check_in")
        )
        self.btn_checkin.pack(fill=tk.X, pady=5)

        self.btn_checkout = tk.Button(
            btn_frame,
            text="📤 LẤY ĐỒ",
            font=btn_font,
            bg="#2980B9",
            fg="white",
            activebackground="#3498DB",
            activeforeground="white",
            relief=tk.RAISED,
            bd=2,
            padx=10,
            pady=10,
            command=lambda: self.process_scan("check_out")
        )
        self.btn_checkout.pack(fill=tk.X, pady=5)

        # ===== ÔI THÔNG TIN NGƯỜI DÙNG =====
        info_frame = tk.LabelFrame(right_frame, text="👤 THÔNG TIN NGƯỜI DÙNG", font=header_font, fg="#00D4FF", bg="#252525", labelanchor="nw", bd=2, relief=tk.RAISED)
        info_frame.pack(fill=tk.X, pady=(0, 15))

        # Tuổi
        tk.Label(info_frame, text="Tuổi:", font=msg_font, fg="#ECF0F1", bg="#252525").pack(anchor=tk.W, padx=10, pady=(10, 2))
        self.lbl_age = tk.Label(info_frame, text="--", font=tkfont.Font(family="Helvetica", size=16, weight="bold"), fg="#00D4FF", bg="#252525")
        self.lbl_age.pack(anchor=tk.W, padx=10, pady=(0, 10))

        # Giới tính
        tk.Label(info_frame, text="Giới tính:", font=msg_font, fg="#ECF0F1", bg="#252525").pack(anchor=tk.W, padx=10, pady=(0, 2))
        self.lbl_gender = tk.Label(info_frame, text="--", font=tkfont.Font(family="Helvetica", size=14), fg="#00D4FF", bg="#252525")
        self.lbl_gender.pack(anchor=tk.W, padx=10, pady=(0, 10))

        # ===== ÔI THÔNG BÁO =====
        msg_frame = tk.LabelFrame(right_frame, text="📢 THÔNG BÁO", font=header_font, fg="#00D4FF", bg="#252525", labelanchor="nw", bd=2, relief=tk.RAISED)
        msg_frame.pack(fill=tk.BOTH, expand=True)

        self.lbl_message = tk.Label(
            msg_frame,
            text="Hệ thống đang khởi động...",
            font=msg_font,
            fg="#F1C40F",
            bg="#252525",
            wraplength=280,
            justify="center",
            relief=tk.SUNKEN,
            bd=1,
            padx=10,
            pady=15
        )
        self.lbl_message.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

    def init_models(self):
        self.show_message("Đang nạp AI Models...", "warning")
        try:
            self.face_detector = FaceDetector()
            self.session = load_model(self.WEIGHTS_PATH)
            self.age_gender_session = load_age_gender_model(self.AGE_GENDER_WEIGHTS)
            self.show_message("✅ Hệ thống sẵn sàng!\nVui lòng nhìn vào camera.", "success")
        except Exception as e:
            self.show_message(f"❌ Lỗi nạp model:\n{str(e)}", "error")

    def show_message(self, text, msg_type="info"):
        color_map = {
            "info": "#ECF0F1",
            "success": "#2ECC71",
            "warning": "#F1C40F",
            "error": "#E74C3C"
        }
        self.msg_queue.put((text, color_map.get(msg_type, "#ECF0F1")))

    def show_user_info(self, age, gender_label, gender_conf):
        self.user_info_queue.put((age, gender_label, gender_conf))

    def process_queue(self):
        # Xử lý thông báo
        try:
            while True:
                text, color = self.msg_queue.get_nowait()
                self.lbl_message.config(text=text, fg=color)
        except queue.Empty:
            pass

        # Xử lý thông tin người dùng
        try:
            while True:
                age, gender_label, gender_conf = self.user_info_queue.get_nowait()
                self.lbl_age.config(text=f"{age:.1f} tuổi")
                self.lbl_gender.config(text=f"{gender_label}\n(Độ tin cậy: {gender_conf:.1%})")
        except queue.Empty:
            pass

        self.window.after(100, self.process_queue)

    def update_frame(self):
        ret, frame = self.vid.read()
        if ret:
            frame = cv2.flip(frame, 1)
            
            # QUAN TRỌNG: Resize frame về size của canvas để fix lệch camera
            frame = cv2.resize(frame, (800, 600))
            
            display_frame = frame.copy()
            ih, iw, _ = frame.shape
            
            self.current_face_img = None

            if self.face_detector is not None:
                detection = self.face_detector.detect(frame)
                if detection is not None:
                    x, y, w, h = detection
                    pad_x = int(w * 0.2)
                    pad_y = int(h * 0.2)

                    x1 = max(0, x - pad_x)
                    y1 = max(0, y - pad_y)
                    x2 = min(iw, x + w + pad_x)
                    y2 = min(ih, y + h + pad_y)

                    if w > 10 and h > 10:
                        self.current_face_img = frame[y1:y2, x1:x2]
                        # Khung xanh khi phát hiện khuôn mặt
                        cv2.rectangle(display_frame, (x, y), (x + w, y + h), (0, 255, 0), 3)
                        cv2.putText(display_frame, "Face Detected", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            # Chuyển RGB để hiển thị trên Tkinter
            cv2image = cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(cv2image)
            imgtk = ImageTk.PhotoImage(image=img)
            
            self.canvas.imgtk = imgtk
            self.canvas.create_image(0, 0, anchor=tk.NW, image=imgtk)

        self.window.after(15, self.update_frame)

    def process_scan(self, action):
        if self.session is None or self.age_gender_session is None:
            self.show_message("❌ Hệ thống AI chưa sẵn sàng!", "error")
            return

        if self.current_face_img is None or self.current_face_img.size == 0:
            self.show_message("❌ Không tìm thấy khuôn mặt.\nVui lòng nhìn thẳng camera.", "error")
            return

        # Trích xuất embedding
        input_array = preprocess_face(self.current_face_img)
        current_vector = get_embedding(self.session, input_array)

        if action == "check_in":
            existing_locker, sim = find_existing_locker(current_vector, self.lockers, self.RECOGNITION_THRESHOLD)
            if existing_locker is not None:
                self.show_message(f"⚠️ Bạn đang có đồ ở tủ #{existing_locker}\n(Độ khớp: {sim:.2f})\nVui lòng lấy đồ ra trước.", "warning")
            else:
                empty_locker = next((lid for lid, vec in self.lockers.items() if vec is None), None)
                if empty_locker is None:
                    self.show_message("❌ Rất tiếc, tất cả các tủ đều đã đầy.", "error")
                else:
                    # Predict age & gender
                    age, gender_label, gender_conf, face_rgb = predict_age_gender(self.age_gender_session, self.current_face_img)
                    
                    # Lưu vector
                    self.lockers[empty_locker] = current_vector
                    
                    # Hiển thị thông tin
                    self.show_user_info(age, gender_label, gender_conf)
                    self.show_message(f"✅ CẤT ĐỒ THÀNH CÔNG!\n\n🔓 Tủ #{empty_locker} đang mở.", "success")

        elif action == "check_out":
            matched_locker, highest_sim = find_existing_locker(current_vector, self.lockers, self.RECOGNITION_THRESHOLD)
            if matched_locker is not None:
                self.lockers[matched_locker] = None
                self.show_message(f"✅ XÁC THỰC THÀNH CÔNG!\n\n🔓 Tủ #{matched_locker} đang mở.\n(Độ khớp: {highest_sim:.2f})", "success")
                # Reset thông tin người dùng
                self.lbl_age.config(text="--")
                self.lbl_gender.config(text="--")
            else:
                self.show_message("❌ Khuôn mặt không khớp với\nbất kỳ tủ nào đang sử dụng.", "error")

    def __del__(self):
        if hasattr(self, 'vid') and self.vid.isOpened():
            self.vid.release()


if __name__ == "__main__":
    root = tk.Tk()
    app = SmartLockerGUI(root, "🔐 Hệ Thống Tủ Đồ Thông Minh")
    root.mainloop()