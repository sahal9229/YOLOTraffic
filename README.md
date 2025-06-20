# Vehicle Detection & Tracking Solution

![Vehicle Detection Banner](https://raw.githubusercontent.com/your-username/your-repo/main/images/banner.jpg)

## Overview

This project provides an optimized framework for detecting and tracking vehicles (cars and bikes) in video streams using the YOLO (You Only Look Once) object detection model, OpenCV for visualization, and custom tracking logic implemented in Python. It enables real-time monitoring, counting, and statistical analysis of vehicles in a video file or webcam feed.

---

## 🚀 Key Strengths

### ✅ Robust Tracking

* Efficiently manages object IDs.
* Matches detections across frames using IoU and spatial proximity.
* Validates vehicles by requiring presence in multiple frames before confirming.

### ✅ Confidence & Confirmation Logic

* Vehicles are confirmed after being detected in **3 consecutive frames**.
* Helps suppress spurious detections and duplicate IDs.

### ✅ Comprehensive Counting

* Tracks:

  * Current visible vehicles
  * Maximum vehicles simultaneously tracked
  * Total unique confirmed vehicles

### ✅ User Interaction

* Live preview
* Pause/Resume
* Screenshot capture
* Tracking reset

### ✅ Code Structure

* Modular design with detection, tracking, and visualization separated cleanly.

### ✅ Performance Optimization

* Uses efficient NumPy operations.
* Achieves near real-time processing.

---

## 📈 Potential Improvements

* **Code Formatting**: Ensure clear spacing and indentation.
* **Generalization**: Allow dynamic detection for all model classes, not just cars and bikes.
* **Documentation**: Add inline comments for better clarity.
* **Resource Release**: Confirm video and window resources are released safely.

---

## 🧠 Code Explanation

### 1️⃣ Detector & Tracker Setup

* **Model Loading**: Loads a YOLO model trained on vehicle data.
* **Video Input**: Accepts both video file and webcam input.

### 2️⃣ The `VehicleTracker` Class

* **Initialization**: Manages visible and total tracked vehicles, and assigns unique IDs.
* **Distance & IoU**:

  * `calculate_distance`: Computes Euclidean distance between box centers.
  * `calculate_iou`: Computes intersection-over-union.
* **Update Logic**:

  * **Pass 1**: Match new detections with existing tracks.
  * **Pass 2**: Unmatched detections become new unconfirmed tracks.
  * **Pass 3**: Preserve temporarily missing vehicles.
  * **Pass 4**: Confirm vehicles with consistent detections.
* **Statistics**:

  * `get_current_counts`
  * `get_total_unique_vehicles`
  * `get_total_by_class`

### 3️⃣ Detection Loop

* Uses YOLOv8 to detect objects in each frame.
* Filters detections to only cars and bikes.
* Updates tracker with each detection.
* Draws:

  * Bounding boxes (color-coded)
  * Vehicle IDs & Confidence
  * Live overlay with current/max/total stats
* User Controls:

  * `q` / `ESC`: Quit
  * `p`: Pause
  * `s`: Save Screenshot
  * `r`: Reset tracking

### 4️⃣ User Interface

* CLI prompts:

  * **Option 1**: Use default video path (hardcoded)
  * **Option 2**: Use live webcam
  * **Option 3**: Enter custom file path (e.g., `C:\Users\sahal\Downloads\video.mp4`, no quotes)

---

## 🌟 Notable Aspects

| Feature              | Description                                                       |
| -------------------- | ----------------------------------------------------------------- |
| Real-Time Capability | Processes at live frame rate (if hardware permits)                |
| Extensibility        | Can support more classes, streams, or custom logic                |
| Application Areas    | Traffic monitoring, parking analytics, surveillance, smart cities |

---

## 📊 Summary Table

| Component            | Functionality                                          |
| -------------------- | ------------------------------------------------------ |
| YOLO Detection       | Identifies vehicles in each video frame                |
| VehicleTracker       | Assigns IDs, tracks vehicles, filters false detections |
| Visualization        | Draws bounding boxes, counters, overlays               |
| User Controls        | Pause, quit, screenshot, reset                         |
| Statistics Reporting | Reports current, max, and total unique vehicles        |

---

## 🖼️ Output Example

![Screenshot 2025-06-20 103748](https://github.com/user-attachments/assets/53434fbb-ed5f-4f39-bcc4-33cd5770d72a)



## 🧪 Output Mode Options
![Screenshot 2025-06-20 102729](https://github.com/user-attachments/assets/93397c42-8662-4ca1-ab10-9d6745165ffe)

1. **Default Video File**: Uses predefined path from the code.
2. **Webcam Feed**: Live detection from your webcam.
3. **Manual Input**: Paste your own video file path without quotes. Example:
   (C:\Users\sahal\Downloads\video.mp4)


---

## 💡 Final Notes

* Detection speed and accuracy depend on your hardware and model.
* You can expand this project with:

  * Speed estimation
  * Entry/Exit line counting
  * Dashboard integration (Streamlit, Flask)

> Designed and built for real-time, extensible vehicle detection solutions 🚗🏍️
