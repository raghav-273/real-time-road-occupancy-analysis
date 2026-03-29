# 🚦 Real-Time Road Occupancy Analysis

## 📌 Overview
This project presents a **real-time traffic monitoring system** that combines classical computer vision and deep learning to analyze road conditions and detect illegal roadside vehicle occupancy.

The system uses **lane detection** to estimate the drivable region and **YOLOv8-based object detection** to identify vehicles. A custom metric, the **Road Occupancy Index (ROI)**, is introduced to quantify road obstruction in real time.

---

## 🎯 Key Features
  - 🚗 Real-time vehicle detection using YOLOv8  
  - 🛣️ Lane detection using edge detection and Hough Transform  
  - 📊 Road Occupancy Index (ROI) for quantitative analysis  
  - 🚫 Detection of vehicles outside lane boundaries  
  - ⚡ Real-time performance with FPS monitoring  
  - 📉 Temporal smoothing for stable output  

---

## 🧠 System Pipeline
```bash
Input Video → Lane Detection → Vehicle Detection (YOLOv8) → Spatial Analysis → ROI Calculation → Visualization
```
---

## 📐 Road Occupancy Index (ROI)

The system introduces a custom metric:

ROI = (Occupied Vehicle Area / Lane Area) × 100

Where:
- Illegal Vehicle Area = area of vehicles outside lane boundaries  
- Lane Area = estimated drivable region  

---

## 🛠️ Tech Stack
- Python  
- OpenCV  
- YOLOv8 (Ultralytics)  
- NumPy  

---

## ⚙️ How It Works

1. Detect lane boundaries using classical CV techniques  
2. Detect vehicles using YOLOv8  
3. Classify vehicles based on lane position  
4. Compute Road Occupancy Index  
5. Apply smoothing for stability  
6. Display results in real time  

---

## 🚀 Getting Started

### 1️⃣ Clone the repository
```bash
git clone https://github.com/raghav-273/real-time-road-occupancy-analysis.git
cd real-time-road-occupancy-analysis
```
### 2️⃣ Install dependencies
```bash
pip install -r requirements.txt
```

### 3️⃣ Run the project
```bash
python src/main.py
```

---

## 📊 Performance
	•	Real-time performance: ~18–30 FPS (CPU)
	•	Stable ROI using temporal smoothing
	•	Lightweight pipeline suitable for real-time systems

___

## ⚠️ Limitations
	•	Lane detection may fail on curves or poor markings
	•	ROI is an approximate geometric metric
	•	No temporal tracking for stationary vehicles

---

## 🔮 Future Improvements
	•	Vehicle tracking (DeepSORT)
	•	Deep learning-based lane detection (segmentation models)
	•	Bird’s-eye view transformation
	•	GPU acceleration for improved performance

---

## 📸 Demo

Demo video available locally or upon request.

---

## 📁 Project Structure
```bash
real-time-road-occupancy-analysis/
│
├── src/                          # Core source code
│   ├── main.py
│   ├── lane_detection.py
│   ├── vehicle_detection.py
│   └── utils.py
│
├── data/                         # (NOT pushed to GitHub)
│   └── road2.mp4
│
├── output/                       # (NOT pushed)
│   └── demo_output.mp4
│
├── models/                       # (NOT pushed)
│   └── yolov8n.pt
│
├── requirements.txt              # Dependencies
├── README.md                     # Project description
├── .gitignore                    # Ignore unnecessary files
└── report.pdf                    # IEEE format
```

---

## 🧠 Key Learning Outcomes
	•	Integration of classical CV and deep learning
	•	Real-time system design
	•	Geometric reasoning for spatial analysis
	•	Designing custom evaluation metrics
  
---

## 👨‍💻 Author

**Raghav Mishra**
BTech CSE (AI)
