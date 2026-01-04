# 🦅 EagleEye: FSCE Recognition & Surveillance System

An advanced real-time surveillance system featuring Face Recognition, Multi-Camera Tracking, GPS Geolocation, and Forensic 3D Mesh Capture.

## ✨ Features
*   **Real-Time Face Recognition**: Uses InsightFace (Buffalo_L) for high-accuracy detection.
*   **Forensics Capture**: Automatically saves:
    *   Original Face Crop (`_orig.jpg`)
    *   **468-Point Mesh** Visualization (MediaPipe) (`_mesh.jpg`)
    *   Enrolled Reference Mesh (`_ref_mesh.jpg`)
*   **Geo-Tracking**: Tracks targets across cameras with GPS coordinates.
*   **Privacy Inpainting**: Occludes non-target faces using LaMa/OpenCV.
*   **Centralized Dashboard**: Web interface for live monitoring and downloading forensic data.

---

## 🚀 Installation Guide

### 1. Prerequisites
*   **OS**: Windows 10/11 (Recommended) or Linux.
*   **Python**: 3.10 or 3.11.
*   **C++ Build Tools**: Required for compiling InsightFace components.
    *   *Windows*: Install "Desktop development with C++" via Visual Studio Build Tools.

### 2. Setup
Clone the repository and install dependencies:

```bash
# 1. Clone
git clone https://github.com/Goutham-P-S/fsce_recog_surv_sys.git
cd fsce_recog_surv_sys

# 2. Virtual Environment (Optional but Recommended)
python -m venv venv
# Windows:
.\venv\Scripts\activate
# Linux:
source venv/bin/activate

# 3. Install Dependencies
pip install -r requirements.txt
```

### 3. Configuration
Create a `.env` file in the root directory if you need customization (optional):
```bash
cp .env.example .env
```

To configure cameras, open `settings.yaml` and edit the `cameras` section:
```yaml
cameras:
  - id: "cam_01"
    source: 0       # Webcam ID or RTSP URL
    gps:
      lat: 19.8776
      lng: 75.3423
```

---

## 🏃 Usage

### Step 1: Enroll Known Faces
You must enroll at least one face to recognize them.
```bash
python tools/enroll_face.py --name "JohnDoe"
```
*   Press **SPACE** to capture photos.
*   Press **Q** to save and exit.

### Step 2: Run Surveillance System
Start the main application and dashboard:
```bash
python main.py
```

### Step 3: Monitor
Open your browser and navigate to:
👉 **[http://localhost:5000](http://localhost:5000)**

*   **Live Stream**: View real-time feeds with overlays.
*   **Logs**: See detection history.
*   **Forensics**: Click [Orig], [Mesh], or [Ref] to download captured evidence.

---

## 🛠 Troubleshooting
*   **Missing DLL / Error 126**: If you see "cublasLt64_12.dll missing", it means you don't have NVIDIA CUDA installed. The system will automatically fallback to CPU mode. This is normal.
*   **MediaPipe Error**: If you get import errors, ensure you are using the pinned version:
    `pip install mediapipe==0.10.9`

## 📦 Building Executable
To package as a standalone `.exe`:
```bash
python tools/build_exe.py
```
See `PACKAGING.md` for details.
