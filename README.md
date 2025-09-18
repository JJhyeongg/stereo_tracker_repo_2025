# Stereo Tracker Repository
Real-Time Robust 2.5D Stereo Multi-Object Tracking with Lightweight Stereo Matching Algorithm

## Installation & Setup

### 1. Create Conda Environment
```bash
conda create -n stereo_tracker python=3.8 -y
conda activate stereo_tracker
```

### 2. Install PyTorch  

- **GPU (CUDA 11.8)**
  ```bash
  pip install torch==2.0.0 torchvision==0.15.1 torchaudio==2.0.1 \
      --index-url https://download.pytorch.org/whl/cu118
  ```

- **CPU only**
  ```bash
  pip install torch==2.0.0 torchvision==0.15.1 torchaudio==2.0.1 \
      --index-url https://download.pytorch.org/whl/cpu
  ```

### 3. Install Other Requirements
```bash
pip install -r requirements.txt
```
---

## Usage

Run stereo tracking with the provided example videos:

```bash
python scripts/track.py \
  --left_src ./videos/example1/rL.mp4 \
  --right_src ./videos/example1/rR.mp4 \
  --params_yaml ./configs/config.yaml \
  --model_path ./models/yolo11_detector.pt \
  --visualize true \
  --save_video ./outputs/stacked.mp4
```

- Press `ESC` to stop visualization.  
- Results will be saved to the path specified by `--save_video`.  

**Command-line options:**
- `--left_src`, `--right_src` : Input video paths (or camera index).  
- `--params_yaml` : Stereo calibration & tracker configuration.  
- `--model_path` : Path to YOLO model.  
- `--visualize` : Whether to display results (`true`/`false`).
- `--visualize_3d` : Enable 3D trajectory plot visualization (default: `false`).  
   - If `true`, 2D visualization (`--visualize`) is automatically enabled.    
- `--save_video` : Path to save visualization video (empty string = no save).  

---

## Project Structure
```
stereo_tracker_repo_2025/
├── configs/
│   ├── config.yaml
│   └── config_manual.yaml
├── models/
│   └── yolo11_detector.pt
├── scripts/
│   ├── stereo_tracker.py
│   └── track.py
└── videos/
    ├── example1/
    │   ├── rL.mp4
    │   └── rR.mp4
    └── example2/
        ├── rL.mp4
        └── rR.mp4
```

---
```
⚠️ **Note:** PyTorch is not included in `requirements.txt` because installation depends on your hardware (GPU vs CPU). Install it manually as described above.