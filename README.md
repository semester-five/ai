# Face Locker - AI Service

Welcome to the AI Service component for the **Face Locker** project. This component encompasses the intelligent core of the smart locker system, allowing users to securely store and retrieve their belongings using facial recognition combined with liveness detection (anti-spoofing).

## 🚀 Features

- **Facial Recognition Check-in / Check-out**: Users can check in (store items) and check out (retrieve items) seamlessly using their face.
- **Accurate & Lightweight**: Uses MobileFaceNet (quantized to INT8 in ONNX format) for rapid, accurate face embedding extraction and cosine similarity matching.
- **Anti-Spoofing (Liveness Detection)**: Integrates a PyTorch-based model to prevent presentation attacks (e.g., using a photo or video to spoof the system), ensuring that the person in front of the kiosk is a real, live human.
- **Real-time Interaction**: Performs object detection (Face Detection) and identity verification directly through the kiosk's webcam in real-time.

## 🛠️ Prerequisites

To run this project, make sure you have Python installed. You will need the following libraries:

- `opencv-python` (for webcam interaction and image processing)
- `numpy` (for heavy array and matrix operations)
- `onnxruntime` (for running the ONNX quantized MobileFaceNet model)
- `torch` & `torchvision` (for the anti-spoofing liveness verification neural network)

*(Note: Install these via your preferred package manager, for instance by running `pip install opencv-python numpy onnxruntime torch torchvision`)*

## 📑 File Structure Highlights

- **`inference.py`**: The baseline kiosk simulation script with basic face recognition Check-In & Check-Out.
- **`inference_spoof.py`**: The advanced kiosk simulation script. Aside from standard face recognition, it employs anti-spoofing to reject counterfeit faces before completing the lock/unlock request.
- **`models/`**: Scripts and definitions regarding models (Face Detector, Anti-Spoof Net).
- **`saved_models/`** *(requires external weights files)*: Folder to place pretrained weights, such as `mobilefacenet_int8.onnx` and `anti_spoof.pth`.

## 🎮 How to Run (Detailed Setup)

Follow these detailed steps to successfully configure and run the Face Locker AI service on your local machine:

1. **Clone the Project**: Start by getting the code folder onto your local machine.
   ```bash
   # (Replace with your actual git URL or simply download the zip)
   git clone <repository_url>
   cd ai
   ```

2. **Set up a Virtual Environment (Recommended)**: This ensures that project dependencies do not interfere with your system-wide Python Python installation.
   ```bash
   python -m venv venv
   
   # Activate on Windows:
   venv\Scripts\activate
   
   # Activate on Linux / MacOS:
   source venv/bin/activate
   ```

3. **Install Dependencies**: Install all required packages for computer vision and deep learning inference.
   ```bash
   pip install opencv-python numpy onnxruntime torch torchvision
   # If a detailed requirements.txt is provided later, you can run:
   # pip install -r requirements.txt
   ```

4. **Verify your Webcam Configuration**: Make sure your system's camera is plugged in and accessible. The scripts default to `cv2.VideoCapture(0)`. Make sure no other application (e.g., Zoom, OBS) is currently hoarding your camera feed.

5. **Setup Pretrained Weights**: The AI models need pre-trained network weights to perform inference. Ensure you copy/download the `.onnx` and `.pth` weight files and correctly place them in the `saved_models/` directory:
   - `saved_models/mobilefacenet_int8.onnx`
   - `saved_models/anti_spoof.pth`
   
   *(If you encounter errors about missing files, double check exactly this step).*

6. **Start the Service**: Launch one of the inference scripts from your terminal, depending on your preferred level of verification security:
   
   To run **Base Mode** (Core Face Recognition without anti-spoofing):
   ```bash
   python inference.py
   ```

   To run **Secure Mode** (Face Recognition + Liveness Anti-Spoofing):
   ```bash
   python inference_spoof.py
   ```

7. **Interact via the Kiosk Output Window**:
   Once running, you should see an active webcam window pop up on your screen.
   - Ensure you are looking directly into the webcam and your face is visible within the green box.
   - **Click on the video window** to ensure it's in focus.
   - Press **`I`** to **Check-In/Store Item**: The system will extract your facial layout and allocate an available locker.
   - Press **`O`** to **Check-Out/Retrieve Item**: The system will authorize your live face, browse the active locker list, and unlock your designated locker.
   - Press **`Q`** to **Quit** to close the camera stream and shut down the Python script safely.

## ⚙️ How It Works

1. **Detection**: OpenCV and the FaceDetector model locate the exact rectangular region of a face from the webcam frame.
2. **Liveness Check (`inference_spoof.py`)**: The cropped face is normalized and passed through the `AntiSpoofNet` model (`.pth`). If the score is below the threshold, the operation is blocked.
3. **Encoding**: The verified live face is put through `MobileFaceNet` (`.onnx`) to extract a multi-dimensional embedding numerical vector.
4. **Matching**: When storing, your embedding vector is saved. When retrieving, the newly generated embedding is compared against the database using Cosine Similarity. If the similarity exceeds `0.60` (the recognition threshold limit), the locker action is granted.

---
*Built for the PBL5 Semester - Smart Locker System.*
