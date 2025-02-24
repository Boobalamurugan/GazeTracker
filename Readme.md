# GazeTracker

## Overview
GazeTracker is a deep learning-based gaze estimation system that detects human gaze direction in videos. This project utilizes the **Gaze-LLE (Gaze Target Estimation via Large-Scale Learned Encoders)** model to estimate gaze targets and visualize them in video frames. The implementation leverages **PyTorch**, **RetinaFace**, and **OpenCV** for efficient real-time gaze tracking.

## Features
- **Face Detection**: Uses RetinaFace for face localization.
- **Gaze Estimation**: Predicts gaze direction using `gazelle_dinov2_vitl14_inout`.
- **Visualization**: Draws gaze target points and directional lines on the video.
- **CLI Support**: Run the model on videos with command-line arguments.

## Acknowledgment
This project is based on the research paper:

**"Gaze-LLE: Gaze Target Estimation via Large-Scale Learned Encoders"**

The authors propose a novel method to estimate gaze targets using a large-scale learned encoder model. The key contributions of the paper include:
- Leveraging **DINOv2 ViT-L14** for feature extraction.
- Introducing a **heatmap-based gaze target estimation**.
- Utilizing in/out classification to improve gaze prediction accuracy.

For more details, please refer to the original paper.

## Installation
To set up the environment, install the required dependencies:
```bash
pip install torch torchvision torchaudio
pip install opencv-python numpy pillow retinaface tqdm
```

## Usage
Run the gaze tracker on a video using the command line:
```bash
python main.py --video_path path/to/input.mp4 --output_videopath path/to/output.mp4
```

### Optional Arguments
- `--start_time`: Specify the start time (in seconds) for processing.
- `--duration`: Specify the duration (in seconds) to process.

Example with optional arguments:
```bash
python main.py --video_path input.mp4 --output_videopath output.mp4 --start_time 10 --duration 30
```

## How It Works
1. The model detects faces using **RetinaFace**.
2. The detected face regions are normalized and passed through **Gazelle (DINOv2 ViT-L14)**.
3. The model predicts **gaze heatmaps** and determines if the face is within the frame.
4. The gaze direction and target are visualized on the frames.
5. The processed video is saved as an output file.

## Model and Data
- The model is loaded using `torch.hub.load('fkryan/gazelle', 'gazelle_dinov2_vitl14_inout')`.
- The `heatmap` output is used to estimate gaze target locations.

## Credits
- **Gaze-LLE authors** for their research and model.
- **PyTorch & OpenCV** for deep learning and image processing.
- **RetinaFace** for face detection.

## License
This project is released under the MIT License.



