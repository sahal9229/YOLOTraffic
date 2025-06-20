import cv2
from ultralytics import YOLO
import numpy as np
from collections import defaultdict
import os

class VehicleTracker:
    def __init__(self, max_distance=100, max_frames_missing=30):
        self.vehicles = {}  # Store currently tracked vehicles
        self.all_vehicles_seen = {}  # Store ALL vehicles ever seen
        self.next_id = 0
        self.max_distance = max_distance
        self.max_frames_missing = max_frames_missing
        self.frame_count = 0
        
    def calculate_distance(self, box1, box2):
        """Calculate distance between two bounding box centers"""
        center1 = ((box1[0] + box1[2]) / 2, (box1[1] + box1[3]) / 2)
        center2 = ((box2[0] + box2[2]) / 2, (box2[1] + box2[3]) / 2)
        return np.sqrt((center1[0] - center2[0])**2 + (center1[1] - center2[1])**2)
    
    def calculate_iou(self, box1, box2):
        """Calculate Intersection over Union (IoU) for better matching"""
        # Calculate intersection area
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])
        
        if x2 <= x1 or y2 <= y1:
            return 0.0
        
        intersection = (x2 - x1) * (y2 - y1)
        
        # Calculate union area
        area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
        area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
        union = area1 + area2 - intersection
        
        return intersection / union if union > 0 else 0.0
    
    def update(self, detections):
        """Update vehicle tracking with new detections"""
        self.frame_count += 1
        
        # Create a list to track which detections have been matched
        matched_detections = set()
        updated_vehicles = {}
        
        # First pass: Match detections to existing vehicles
        for detection_idx, detection in enumerate(detections):
            bbox, class_name, confidence = detection
            best_match_id = None
            best_score = 0.0  # Use combined score instead of just distance
            
            # Find best matching existing vehicle
            for vehicle_id, vehicle_data in self.vehicles.items():
                if vehicle_data['class'] == class_name:
                    # Calculate distance
                    distance = self.calculate_distance(bbox, vehicle_data['bbox'])
                    # Calculate IoU
                    iou = self.calculate_iou(bbox, vehicle_data['bbox'])
                    
                    # Combined matching score (distance + IoU)
                    if distance < self.max_distance:
                        # Normalize distance (smaller is better) and combine with IoU (larger is better)
                        distance_score = max(0, 1 - distance / self.max_distance)
                        combined_score = (distance_score * 0.6) + (iou * 0.4)
                        
                        if combined_score > best_score and combined_score > 0.3:  # Minimum threshold
                            best_score = combined_score
                            best_match_id = vehicle_id
            
            # Update existing vehicle or mark for new vehicle creation
            if best_match_id is not None:
                updated_vehicles[best_match_id] = {
                    'bbox': bbox,
                    'class': class_name,
                    'confidence': confidence,
                    'last_seen': self.frame_count,
                    'first_seen': self.vehicles[best_match_id]['first_seen'],
                    'consecutive_detections': self.vehicles[best_match_id].get('consecutive_detections', 0) + 1
                }
                matched_detections.add(detection_idx)
        
        # Second pass: Create new vehicles for unmatched detections
        for detection_idx, detection in enumerate(detections):
            if detection_idx not in matched_detections:
                bbox, class_name, confidence = detection
                
                # Create new vehicle with confirmation requirement
                vehicle_id = self.next_id
                updated_vehicles[vehicle_id] = {
                    'bbox': bbox,
                    'class': class_name,
                    'confidence': confidence,
                    'last_seen': self.frame_count,
                    'first_seen': self.frame_count,
                    'consecutive_detections': 1,
                    'confirmed': False  # Require confirmation before counting
                }
                
                # Store in all_vehicles_seen for permanent record
                self.all_vehicles_seen[vehicle_id] = {
                    'class': class_name,
                    'first_seen': self.frame_count,
                    'last_seen': self.frame_count
                }
                
                self.next_id += 1
        
        # Third pass: Keep vehicles that were seen recently (even if not detected this frame)
        for vehicle_id, vehicle_data in self.vehicles.items():
            if (vehicle_id not in updated_vehicles and 
                self.frame_count - vehicle_data['last_seen'] <= self.max_frames_missing):
                # Decrease consecutive detections for missed frames
                vehicle_data['consecutive_detections'] = max(0, vehicle_data.get('consecutive_detections', 0) - 1)
                updated_vehicles[vehicle_id] = vehicle_data
        
        # Fourth pass: Confirm vehicles that have been detected consistently
        for vehicle_id, vehicle_data in updated_vehicles.items():
            if not vehicle_data.get('confirmed', True):  # If not confirmed yet
                if vehicle_data['consecutive_detections'] >= 3:  # Require 3 consecutive detections
                    vehicle_data['confirmed'] = True
            
            # Update permanent record
            if vehicle_id in self.all_vehicles_seen:
                self.all_vehicles_seen[vehicle_id]['last_seen'] = self.frame_count
        
        self.vehicles = updated_vehicles
        return self.vehicles
    
    def get_current_counts(self):
        """Get count of currently visible confirmed vehicles by class"""
        counts = {'cars': 0, 'bike': 0}
        for vehicle_data in self.vehicles.values():
            if vehicle_data.get('confirmed', True) and vehicle_data['class'] in counts:
                counts[vehicle_data['class']] += 1
        return counts
    
    def get_total_unique_vehicles(self):
        """Get total count of all unique confirmed vehicles ever seen"""
        confirmed_vehicles = 0
        for vehicle_id, vehicle_data in self.all_vehicles_seen.items():
            # Only count vehicles that have been confirmed
            if vehicle_id in self.vehicles and self.vehicles[vehicle_id].get('confirmed', True):
                confirmed_vehicles += 1
            elif vehicle_id not in self.vehicles:
                # Vehicle no longer tracked, assume it was confirmed if it existed long enough
                if vehicle_data['last_seen'] - vehicle_data['first_seen'] >= 2:
                    confirmed_vehicles += 1
        
        return confirmed_vehicles
    
    def get_total_by_class(self):
        """Get total count of confirmed vehicles by class"""
        counts = {'cars': 0, 'bike': 0}
        for vehicle_id, vehicle_data in self.all_vehicles_seen.items():
            # Only count confirmed vehicles
            confirmed = False
            if vehicle_id in self.vehicles:
                confirmed = self.vehicles[vehicle_id].get('confirmed', True)
            else:
                # Vehicle no longer tracked, assume confirmed if existed long enough
                confirmed = vehicle_data['last_seen'] - vehicle_data['first_seen'] >= 2
            
            if confirmed and vehicle_data['class'] in counts:
                counts[vehicle_data['class']] += 1
        
        return counts

def detect_vehicles_optimized(source, live_preview=True, confidence_threshold=0.5):
    """
    Optimized vehicle detection with proper tracking
    """
    
    # Load the YOLO model
    try:
        model = YOLO(r"runs\detect\train\weights\best.pt")
        class_names = model.names
        print(f"Model loaded successfully. Available classes: {list(class_names.values())}")
    except Exception as e:
        print(f"Error loading model: {e}")
        return
    
    # Check if video file exists
    if isinstance(source, str) and source != "0" and not source.isdigit():
        if not os.path.exists(source):
            print(f"Error: Video file '{source}' not found!")
            return
    
    # Load video/webcam
    cap = cv2.VideoCapture(source)
    
    if not cap.isOpened():
        print(f"Error: Cannot open video source '{source}'")
        return
    
    # Get video properties
    fps = int(cap.get(cv2.CAP_PROP_FPS)) or 30
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    print(f"Video info: {width}x{height}, {fps} FPS")
    if total_frames > 0:
        print(f"Total frames: {total_frames}")
    
    # Initialize vehicle tracker with optimized parameters
    tracker = VehicleTracker(max_distance=120, max_frames_missing=20)
    
    # Statistics
    frame_count = 0
    detection_history = []
    max_vehicles_seen = {'cars': 0, 'bike': 0}
    
    print(f"Starting optimized detection...")
    print("Controls: 'q'=quit, 'p'=pause, 's'=screenshot, 'r'=reset counts")
    print("-" * 60)
    
    paused = False
    
    while cap.isOpened():
        if not paused:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_count += 1
            
            # Run YOLO detection with confidence threshold
            results = model(frame, conf=confidence_threshold)[0]
            
            # Extract detections
            detections = []
            if results.boxes is not None:
                for box in results.boxes:
                    cls_id = int(box.cls[0])
                    class_name = class_names[cls_id]
                    conf = float(box.conf[0])
                    bbox = box.xyxy[0].cpu().numpy()  # [x1, y1, x2, y2]
                    
                    # Only track cars and bikes
                    if class_name in ['cars', 'bike']:
                        detections.append((bbox, class_name, conf))
            
            # Update tracker
            tracked_vehicles = tracker.update(detections)
            current_counts = tracker.get_current_counts()
            total_counts = tracker.get_total_by_class()
            
            # Update maximum counts
            for vehicle_type, count in current_counts.items():
                if count > max_vehicles_seen[vehicle_type]:
                    max_vehicles_seen[vehicle_type] = count
            
            # Store detection history
            detection_history.append({
                'frame': frame_count,
                'current_cars': current_counts['cars'],
                'current_bikes': current_counts['bike'],
                'total_detections': len(detections)
            })
            
            # Print significant changes
            if frame_count == 1 or frame_count % 60 == 0:
                total_unique = tracker.get_total_unique_vehicles()
                print(f"Frame {frame_count}: Current - Cars: {current_counts['cars']}, Bikes: {current_counts['bike']}, Total Unique: {total_unique}")
            
            # Draw results on frame
            if live_preview:
                # Draw bounding boxes for tracked vehicles
                for vehicle_id, vehicle_data in tracked_vehicles.items():
                    bbox = vehicle_data['bbox']
                    class_name = vehicle_data['class']
                    conf = vehicle_data['confidence']
                    confirmed = vehicle_data.get('confirmed', True)
                    
                    x1, y1, x2, y2 = map(int, bbox)
                    
                    # Color coding - different colors for confirmed vs unconfirmed
                    if class_name == 'cars':
                        color = (0, 255, 0) if confirmed else (0, 150, 0)  # Green for cars
                        display_name = "Car"
                    else:
                        color = (255, 0, 0) if confirmed else (150, 0, 0)  # Blue for bikes
                        display_name = "Bike"
                    
                    # Draw bounding box - thicker for confirmed vehicles
                    thickness = 3 if confirmed else 1
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)
                    
                    # Draw label with vehicle ID and confirmation status
                    status = "✓" if confirmed else "?"
                    label = f"{display_name} #{vehicle_id} {status} ({conf:.2f})"
                    label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
                    
                    # Background for label
                    cv2.rectangle(frame, (x1, y1 - label_size[1] - 10), 
                                (x1 + label_size[0], y1), color, -1)
                    cv2.putText(frame, label, (x1, y1 - 5), 
                              cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
                
                # Add info panel
                info_panel_height = 150
                overlay = np.zeros((info_panel_height, width, 3), dtype=np.uint8)
                
                # Current counts
                cv2.putText(overlay, f"Frame: {frame_count}", (10, 25), 
                          cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                cv2.putText(overlay, f"Currently Visible:", (10, 50), 
                          cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                cv2.putText(overlay, f"Cars: {current_counts['cars']}", (10, 75), 
                          cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                cv2.putText(overlay, f"Bikes: {current_counts['bike']}", (10, 100), 
                          cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
                
                # Maximum seen simultaneously
                cv2.putText(overlay, f"Max Simultaneous:", (200, 50), 
                          cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                cv2.putText(overlay, f"Cars: {max_vehicles_seen['cars']}", (200, 75), 
                          cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                cv2.putText(overlay, f"Bikes: {max_vehicles_seen['bike']}", (200, 100), 
                          cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
                
                # Total unique vehicles
                total_unique = tracker.get_total_unique_vehicles()
                cv2.putText(overlay, f"Total Unique:", (400, 50), 
                          cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                cv2.putText(overlay, f"All: {total_unique}", (400, 75), 
                          cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
                cv2.putText(overlay, f"Cars: {total_counts['cars']}, Bikes: {total_counts['bike']}", (400, 100), 
                          cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)
                
                # Status indicators
                cv2.putText(overlay, f"✓ = Confirmed, ? = Pending", (10, 130), 
                          cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
                
                # Combine overlay with frame
                frame_with_overlay = np.vstack([overlay, frame])
                cv2.imshow("Fixed Vehicle Detection", frame_with_overlay)
        
        # Handle key presses
        if live_preview:
            if total_frames > 0:
                key = cv2.waitKey(max(1, int(1000/fps))) & 0xFF
            else:
                key = cv2.waitKey(1) & 0xFF
            
            if key == ord('q') or key == 27:  # Quit
                break
            elif key == ord('p'):  # Pause
                paused = not paused
                print("Paused" if paused else "Resumed")
            elif key == ord('s'):  # Screenshot
                screenshot_name = f"detection_frame_{frame_count}.jpg"
                cv2.imwrite(screenshot_name, frame)
                print(f"Screenshot saved: {screenshot_name}")
            elif key == ord('r'):  # Reset tracking
                tracker = VehicleTracker(max_distance=120, max_frames_missing=20)
                max_vehicles_seen = {'cars': 0, 'bike': 0}
                print("Tracking reset!")
    
    # Cleanup
    cap.release()
    cv2.destroyAllWindows()
    
    # Final analysis
    final_total_counts = tracker.get_total_by_class()
    final_unique_total = tracker.get_total_unique_vehicles()
    
    print("\n" + "=" * 60)
    print("FIXED DETECTION SUMMARY")
    print("=" * 60)
    print(f"Total frames processed: {frame_count}")
    print(f"Total unique vehicles detected: {final_unique_total}")
    print(f"Total unique cars: {final_total_counts['cars']}")
    print(f"Total unique bikes: {final_total_counts['bike']}")
    print(f"Maximum cars visible simultaneously: {max_vehicles_seen['cars']}")
    print(f"Maximum bikes visible simultaneously: {max_vehicles_seen['bike']}")
    
    # Calculate average vehicles per frame
    if detection_history:
        avg_cars = sum(d['current_cars'] for d in detection_history) / len(detection_history)
        avg_bikes = sum(d['current_bikes'] for d in detection_history) / len(detection_history)
        print(f"Average cars visible per frame: {avg_cars:.2f}")
        print(f"Average bikes visible per frame: {avg_bikes:.2f}")
    
    # Show frame-by-frame summary for key frames
    print(f"\nDetection Timeline (every 60 frames):")
    for i, data in enumerate(detection_history):
        if i % 60 == 0 or i == len(detection_history) - 1:
            print(f"Frame {data['frame']:4d}: Cars={data['current_cars']}, Bikes={data['current_bikes']}")
    
    print("\nFixed detection completed!")
    print("Note: Only confirmed vehicles (seen for 3+ consecutive frames) are counted")

# Main execution
if __name__ == "__main__":
    # Your video path
    video_path = r"C:\Users\sahal\Downloads\27260-362770008_medium.mp4"
    
    print("Fixed Vehicle Detection System")
    print("=" * 40)
    print("1. Process your video file (recommended)")
    print("2. Use webcam (live)")
    print("3. Use different video file")
    
    choice = input("\nEnter choice (1/2/3): ").strip()
    
    if choice == "1":
        print(f"\nProcessing video: {video_path}")
        detect_vehicles_optimized(source=video_path, live_preview=True, confidence_threshold=0.6)
    
    elif choice == "2":
        print("\nStarting webcam detection...")
        detect_vehicles_optimized(source=0, live_preview=True, confidence_threshold=0.5)
    
    elif choice == "3":
        video_file = input("Enter full path to video file: ").strip()
        detect_vehicles_optimized(source=video_file, live_preview=True, confidence_threshold=0.6)
    
    else:
        print("\nInvalid choice. Processing your video file...")
        detect_vehicles_optimized(source=video_path, live_preview=True, confidence_threshold=0.6)