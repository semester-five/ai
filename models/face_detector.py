import cv2

# Face Detector 
class FaceDetector:
    """
    Dùng OpenCV's built-in Haar Cascade.
    """
    def __init__(self):
        # Haar Cascade — model để khoanh vùng khuôn mặt trong hình
        self.detector = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    def detect(self, frame):
        """Trả về (x, y, w, h) của khuôn mặt đầu tiên tìm thấy, hoặc None."""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.detector.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(80, 80)   # Bỏ qua khuôn mặt quá nhỏ
        )
        if len(faces) == 0:
            return None

        # Lấy khuôn mặt lớn nhất (gần camera nhất)
        faces = sorted(faces, key=lambda f: f[2] * f[3], reverse=True)
        return faces[0]  # (x, y, w, h)