


# Drowning Detection System

This is a real-time drowning detection system built using YOLOv3 and PyTorch. It uses computer vision techniques to identify people in water and trigger alerts when potential drowning is detected.

## Features

- Real-time object detection using YOLOv3
- Customizable drowning detection logic
- Audio or visual alerts on detection
- Modular, readable code structure for easy development

## Getting Started

### 1. Clone the Repository

```bash
git clone https://github.com/Samaspp/Drowning-Detection.git
cd Drowning-Detection
````

### 2. Set Up the Environment

Create and activate a virtual environment (optional but recommended):

```bash
python -m venv venv
source venv/bin/activate      # On Windows: venv\Scripts\activate
```

Then install dependencies:

```bash
pip install -r requirements.txt
```

### 3. Run the Application

```bash
python app.py
```

Or open the notebook:

```bash
jupyter notebook test.ipynb
```

## Model Files

Make sure the following model-related files are present in the root directory:

* `model.pth` — the trained PyTorch model used for detection
* `yolov3.cfg` — YOLOv3 configuration
* `yolov3.txt` — class names for YOLOv3

Note: `yolov3.weights` is not included due to GitHub’s size limits. If needed, it should be downloaded separately or converted into a smaller `.pth` format.

## Project Structure

```
Drowning-Detection/
├── app.py
├── test.ipynb
├── requirements.txt
├── model.pth
├── yolov3.cfg
├── yolov3.txt
│
├── cvlib/
│   ├── object_detection.py
│   ├── utils.py
│   ├── data/
│   └── resources/
│
├── output/              # Stores generated logs/images (optional)
└── sound/               # Audio alerts
```

## Contribution

If you're interested in improving this system, feel free to fork the repo and submit a pull request. For major changes, open an issue first to discuss the proposal.

