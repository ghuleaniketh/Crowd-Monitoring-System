# test_model.py - Fixed for webcam detection
from ultralytics import YOLO
import cv2

def find_webcam():
    """Find working webcam index"""
    print("🔍 Searching for your webcam...")
    
    for i in range(5):  # Check indices 0-4
        print(f"   Testing camera {i}...")
        cap = cv2.VideoCapture(i)
        
        if cap.isOpened():
            ret, frame = cap.read()
            if ret and frame is not None:
                print(f"✅ Found webcam at index {i}")
                return cap, i
            cap.release()
        
    print("❌ No webcam found!")
    return None, None

# Load YOLOv8 human detection model
print("📦 Loading YOLOv8 human detection model...")
try:
    model = YOLO('best.pt')
    print("✅ YOLOv8 model loaded successfully!")
except Exception as e:
    print(f"❌ Error loading model: {e}")
    print("Make sure 'best.pt' file is in the same folder")
    exit()

# Find and connect to webcam
cap, camera_index = find_webcam()
if cap is None:
    print("❌ Cannot find any working webcam")
    print("💡 Make sure your webcam is connected and not used by other apps")
    exit()

print(f"🎥 Using webcam at index {camera_index}")
print("📹 Starting human detection...")
print("Press 'q' to quit")

# Human detection loop
while True:
    ret, frame = cap.read()
    if not ret:
        print("❌ Cannot read from webcam")
        break
    
    # Run YOLOv8 human detection
    results = model(frame, verbose=False)  # verbose=False reduces console spam
    
    # Count detected humans
    human_count = 0
    if results[0].boxes is not None:
        for box in results[0].boxes:
            # Check if detected object is a person (class 0 in COCO dataset)
            if int(box.cls) == 0:  # Person class
                human_count += 1
    
    # Draw detection results
    annotated_frame = results[0].plot()
    
    # Add human count display
    cv2.putText(annotated_frame, f'Humans Detected: {human_count}', 
               (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    
    cv2.putText(annotated_frame, f'Camera: {camera_index} | Press Q to quit', 
               (10, annotated_frame.shape[0] - 10), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    # Display result
    cv2.imshow('YOLOv8 Human Detection', annotated_frame)
    
    # Print detection info
    if human_count > 0:
        print(f"👤 Detected {human_count} human(s)")
    
    # Quit on 'q' key
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Cleanup
cap.release()
cv2.destroyAllWindows()
print("✅ YOLOv8 human detection test completed!")
