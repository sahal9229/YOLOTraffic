# Vehicle Detection & Tracking Solution

## Overview

This project provides an optimized framework for detecting and tracking vehicles (cars and bikes) in video streams using the YOLO (You Only Look Once) object detection model, OpenCV for visualization, and custom tracking logic implemented in Python. It enables real-time monitoring, counting, and statistical analysis of vehicles in a video file or webcam feed.

## Key Strengths

### Robust Tracking

* The `VehicleTracker` class efficiently manages object IDs.
* Matches detections across frames by combining IoU (Intersection over Union) and spatial proximity.
* Vehicles are validated only after being seen in multiple frames, reducing false positives.

### Confidence & Confirmation Logic

* Vehicles are "confirmed" only after being detected for 3 consecutive frames.
* Suppresses duplicate IDs and spurious detections.

### Comprehensive Counting

* Differentiates between:

  * Current visible vehicles
  * Maximum simultaneously tracked
  * Total unique (confirmed) vehicles
* Offers granular insights into vehicle flow and density.

### User Interaction

* Supports live preview, pausing, screen capture, and real-time reset of tracking.
* User-friendly for analysis and debugging.

### Code Structure

* Modular and organized.
* Clear separation between detection, tracking, and visualization.

### Performance Optimization

* Efficient use of NumPy operations.
* Capable of near real-time frame-by-frame analysis.

## Potential Improvements

* **Code Formatting**: Improve spacing and indentation for better readability.
* **Generalization**: Adapt logic to dynamically support all classes in the YOLO model.
* **Documentation**: Add more inline comments, especially in core tracking functions.
* **Resource Management**: Ensure all video and window resources are closed properly.

## Code Explanation

### 1. Detector & Tracker Setup

* **Model Loading**: Loads a trained YOLO model. Prints error if failed.
* **Video Input**: Supports webcam and video file. Extracts FPS, dimensions.

### 2. The VehicleTracker Class

* **Initialization**: Tracks visible vehicles, maintains all seen vehicles, assigns unique IDs.
* **Distance & IoU**:

  * `calculate_distance`: Euclidean distance between centers of boxes.
  * `calculate_iou`: Measures overlap between boxes.
* **Update Logic**:

  * Matches new detections with existing tracks.
  * Unmatched detections are tracked as new "unconfirmed".
  * Tracks that temporarily disappear are preserved.
  * Vehicles are marked "confirmed" if seen consistently.
* **Statistics**:

  * `get_current_counts`
  * `get_total_unique_vehicles`
  * `get_total_by_class`

### 3. Detection Loop

* **Detection Execution**: YOLO detects vehicles per frame.
* **Filtering**: Considers only 'car' and 'bike' classes.
* **Tracker Update**: Each detection updates the tracker.
* **Visualization**:

  * Bounding boxes with colors per class.
  * Overlay section shows stats.
* **Controls**: Pause (`p`), Quit (`q` or `ESC`), Screenshot (`s`), Reset (`r`).

### 4. Statistical Output

* At video end:

  * Prints total unique vehicles.
  * Per-class counts.
  * Max simultaneous counts.
  * Timeline snapshot of detections.

### 5. User Interface

* **CLI Prompt**:

  * Option 1: Use default video link in code.
  * Option 2: Use webcam for live detection.
  * Option 3: Manually input video file path (without quotes).

## Notable Aspects

* **Real-Time Capability**: Near real-time on capable hardware.
* **Extensibility**: Can expand to other object types or streams.
* **Applications**: Useful for:

  * Traffic analysis
  * Parking management
  * Surveillance
  * Smart city planning

## Summary Table

| Component            | Function                                                    |
| -------------------- | ----------------------------------------------------------- |
| YOLO Detection       | Identifies vehicles in each video frame                     |
| VehicleTracker       | Assigns IDs, tracks across frames, filters false detections |
| Visualization        | Draws bounding boxes, stats overlay                         |
| User Controls        | Pause, quit, screenshot, reset                              |
| Statistics Reporting | Current, max, and total unique vehicle tracking             |

## Output

The following modes are supported:

1. **Default Video File**: Uses the hardcoded video path in the code.
2. **Webcam Mode**: Uses webcam feed as the input.
3. **Manual Video File Input**: Enter your own file path (e.g., `C:\Users\sahal\Downloads\video.mp4`) — *no quotes required*.

---

Feel free to explore and expand this project for your specific needs!
