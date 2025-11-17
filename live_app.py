from flask import Flask, render_template, Response, jsonify
import cv2
import time
from ultralytics import YOLO
import numpy as np
from collections import deque
import math

# Flask app setup
app = Flask(__name__)

# Global variables
current_human_count = 0
total_detections = 0
start_time = time.time()
camera_index = "IP Cam @ 192.168.0.218"
detection_active = True

# Detection tracking variables
current_detections = []
detection_history = deque(maxlen=100)
area_counts = {}
frame_width = 1280
frame_height = 720
confidence_threshold = 0.5
tracking_tolerance = 50

# Load YOLOv8 model
print("📦 Loading YOLOv8 model...")
model = YOLO("best.pt")
print("✅ YOLOv8 model loaded!")

SPORTS_AREAS = {
    # Example area, add all your areas here
    "main_court_center": {"x1": 0.35, "y1": 0.35, "x2": 0.65, "y2": 0.65, "name": "Main Court Center"},
    # Add other areas like:
    # "table_tennis_1": {"x1": ..., "y1": ..., "x2": ..., "y2": ..., "name": "..."},
}


def convert_numpy_types(obj):
    """Recursively convert numpy types to native Python types for JSON serialization."""
    if isinstance(obj, dict):
        return {k: convert_numpy_types(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(i) for i in obj]
    elif isinstance(obj, (np.float32, np.float64)):
        return float(obj)
    elif isinstance(obj, (np.int32, np.int64)):
        return int(obj)
    else:
        return obj


def normalize_coordinates_precise(x, y, img_width, img_height):
    norm_x = float(x) / float(img_width)
    norm_y = float(y) / float(img_height)
    norm_x = max(0.0, min(1.0, norm_x))
    norm_y = max(0.0, min(1.0, norm_y))
    return round(norm_x, 6), round(norm_y, 6)

def get_area_from_coordinates_precise(norm_x, norm_y):
    area_priority = [
        "main_court_center",
        "table_tennis_1", "table_tennis_2", "table_tennis_3",
        "entrance_west_north", "entrance_west_south",
        "entrance_east_north", "entrance_east_south",
        "main_court_north", "main_court_south", "main_court_east", "main_court_west",
        "seating_north_left", "seating_north_right",
        "seating_south_left", "seating_south_right",
        "corridor_north", "corridor_south"
    ]
    for area_id in area_priority:
        if area_id in SPORTS_AREAS:
            area_info = SPORTS_AREAS[area_id]
            if (area_info["x1"] <= norm_x <= area_info["x2"] and
                area_info["y1"] <= norm_y <= area_info["y2"]):
                return area_id
    return "other"

def calculate_detection_confidence_score(box, frame_area):
    base_conf = float(box.conf[0])
    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
    box_area = (x2 - x1) * (y2 - y1)
    relative_size = box_area / frame_area

    size_conf = 1.0
    if relative_size < 0.001:
        size_conf = 0.5
    elif relative_size > 0.1:
        size_conf = 0.7

    width = x2 - x1
    height = y2 - y1
    aspect_ratio = height / width if width > 0 else 1
    aspect_conf = min(1.0, aspect_ratio / 2.5) if aspect_ratio > 1 else 0.6

    combined_conf = (base_conf * 0.6) + (size_conf * 0.2) + (aspect_conf * 0.2)
    return round(combined_conf, 4)

def track_detection_movement(new_detections, previous_detections):
    tracked_detections = []
    used_ids = set()

    for new_det in new_detections:
        best_match = None
        min_distance = float('inf')

        for prev_det in previous_detections:
            distance = math.sqrt(
                (new_det['pixel_x'] - prev_det['pixel_x'])**2 +
                (new_det['pixel_y'] - prev_det['pixel_y'])**2
            )
            if distance < min_distance and distance < tracking_tolerance and prev_det.get('tracking_id') not in used_ids:
                min_distance = distance
                best_match = prev_det

        if best_match:
            new_det['tracking_id'] = best_match['tracking_id']
            used_ids.add(best_match['tracking_id'])
            new_det['movement_distance'] = min_distance
            new_det['is_tracked'] = True
        else:
            new_det['tracking_id'] = f"person_{len(tracked_detections)}"
            new_det['movement_distance'] = 0
            new_det['is_tracked'] = False

        tracked_detections.append(new_det)

    return tracked_detections

def update_area_counts_precise():
    global area_counts
    area_counts = {area: 0 for area in SPORTS_AREAS.keys()}
    area_counts["other"] = 0

    for detection in current_detections:
        if detection['confidence'] >= confidence_threshold:
            area = get_area_from_coordinates_precise(detection["x"], detection["y"])
            area_counts[area] += 1

def generate_live_frames_opencv():
    global current_human_count, total_detections, detection_active
    global current_detections, detection_history

    rtsp_url = 'rtsp://NTLCAM:ntlcam123@192.168.0.218:554/stream2'
    cap = cv2.VideoCapture(rtsp_url, cv2.CAP_FFMPEG)

    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    cap.set(cv2.CAP_PROP_FPS, 20)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, frame_width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, frame_height)

    if not cap.isOpened():
        print("❌ Failed to open IP camera stream.")
        return

    print(f"✅ Enhanced OpenCV camera stream opened at {frame_width}x{frame_height} for high-precision tracking.")

    frame_count = 0
    previous_detections = []

    while True:
        if not detection_active:
            time.sleep(0.1)
            continue

        cap.grab()
        success, frame = cap.read()
        if not success:
            continue

        frame_count += 1

        frame = cv2.resize(frame, (frame_width, frame_height))
        frame_area = frame_width * frame_height

        # YOLOv8 detection
        results = model(frame, conf=confidence_threshold)

        new_detections = []
        human_count = 0

        if results[0].boxes is not None:
            for i, box in enumerate(results[0].boxes):
                if int(box.cls) == 0:  # Person class
                    conf_score = calculate_detection_confidence_score(box, frame_area)
                    if conf_score >= confidence_threshold:
                        human_count += 1
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                        center_x = (x1 + x2) / 2.0
                        center_y = (y1 + y2) / 2.0
                        bottom_center_x = center_x
                        bottom_center_y = y2

                        norm_x, norm_y = normalize_coordinates_precise(center_x, center_y, frame_width, frame_height)
                        norm_bottom_x, norm_bottom_y = normalize_coordinates_precise(bottom_center_x, bottom_center_y, frame_width, frame_height)
                        detection_data = {
                            "x": float(norm_x),
                            "y": float(norm_y),
                            "x_bottom": float(norm_bottom_x),
                            "y_bottom": float(norm_bottom_y),
                            "confidence": float(conf_score),
                            "yolo_confidence": float(box.conf[0]),
                            "pixel_x": float(center_x),
                            "pixel_y": float(center_y),
                            "pixel_bottom_x": float(bottom_center_x),
                            "pixel_bottom_y": float(bottom_center_y),
                            "bbox_width": float(x2 - x1),
                            "bbox_height": float(y2 - y1),
                            "bbox_area": float((x2 - x1) * (y2 - y1)),
                            "detection_id": f"det_{frame_count}_{i}",
                            "timestamp": float(time.time())
}


                        new_detections.append(detection_data)

        current_detections = track_detection_movement(new_detections, previous_detections)
        previous_detections = current_detections.copy()

        current_human_count = human_count
        if human_count > 0:
            total_detections += 1

        update_area_counts_precise()

        detection_history.append({
            "timestamp": time.time(),
            "frame_count": frame_count,
            "detections": current_detections.copy(),
            "total_count": human_count,
            "avg_confidence": np.mean([d['confidence'] for d in current_detections]) if current_detections else 0,
            "frame_resolution": f"{frame_width}x{frame_height}"
        })

        annotated_frame = results[0].plot()
        cv2.putText(annotated_frame, f'HIGH-PRECISION YOLOv8 → Heatmap ({frame_width}x{frame_height})',
                    (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(annotated_frame, f'Humans: {human_count} | Precision: 6-decimal | Conf≥{confidence_threshold}',
                    (10, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(annotated_frame, f'Tracking: {sum(1 for d in current_detections if d.get("is_tracked", False))} tracked',
                    (10, 85), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
        cv2.putText(annotated_frame, f'Camera: {camera_index} | Frame: {frame_count}',
                    (10, 115), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        timestamp = time.strftime("%H:%M:%S")
        cv2.putText(annotated_frame, f'{timestamp}',
                    (frame_width - 150, frame_height - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        # Encode frame as JPEG
        ret, jpeg = cv2.imencode('.jpg', annotated_frame)
        if not ret:
            continue

        # Yield frame bytes for streaming
        frame_bytes = jpeg.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

    cap.release()

# Flask routes
@app.route('/')
def home():
    return render_template('live.html')

@app.route('/live_feed')
def live_feed():
    return Response(generate_live_frames_opencv(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/api/live_stats')
def live_stats():
    uptime = int(time.time() - start_time)
    return jsonify({
        'current_humans': current_human_count,
        'total_detections': total_detections,
        'camera_index': camera_index,
        'uptime_seconds': uptime,
        'detection_active': detection_active,
        'timestamp': time.strftime("%Y-%m-%d %H:%M:%S"),
        'area_counts': area_counts,
        'areas_info': SPORTS_AREAS,
        'precision_level': "6-decimal",
        'resolution': f"{frame_width}x{frame_height}",
        'confidence_threshold': confidence_threshold
    })
@app.route('/api/crowd_coordinates')
def crowd_coordinates():
    try:
        safe_detections = convert_numpy_types(current_detections)

        return jsonify({
            'detections': safe_detections,
            'total_count': current_human_count,
            'area_counts': area_counts,
            'timestamp': time.strftime("%Y-%m-%d %H:%M:%S"),
            'precision_info': {
                'coordinate_precision': 6,
                'resolution': f"{frame_width}x{frame_height}",
                'confidence_threshold': confidence_threshold,
                'tracking_enabled': True
            }
        })
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({
            'error': 'Internal server error',
            'message': str(e)
        }), 500


@app.route('/api/area_analytics_precise')
def area_analytics_precise():
    """Get high-precision area-wise crowd analytics"""
    analytics = {}

    for area_id, area_info in SPORTS_AREAS.items():
        area_history = []
        confidence_history = []

        for frame in detection_history:
            area_detections = [det for det in frame['detections']
                               if get_area_from_coordinates_precise(det['x'], det['y']) == area_id]
            count = len(area_detections)
            area_history.append(count)

            if area_detections:
                avg_conf = np.mean([det['confidence'] for det in area_detections])
                confidence_history.append(avg_conf)

        analytics[area_id] = {
            'name': area_info['name'],
            'current_count': area_counts.get(area_id, 0),
            'avg_count': round(np.mean(area_history), 2) if area_history else 0,
            'max_count': max(area_history) if area_history else 0,
            'avg_confidence': round(np.mean(confidence_history), 4) if confidence_history else 0,
            'coordinates': {
                'x1': area_info['x1'], 'y1': area_info['y1'],
                'x2': area_info['x2'], 'y2': area_info['y2']
            }
        }

    return jsonify(analytics)

@app.route('/api/toggle_detection')
def toggle_detection():
    global detection_active
    detection_active = not detection_active
    status = "started" if detection_active else "stopped"
    print(f"🔄 High-precision detection {status}")
    return jsonify({
        'detection_active': detection_active,
        'message': f'High-precision detection {status}'
    })

@app.route('/api/adjust_precision')
def adjust_precision():
    """API to adjust precision settings"""
    global confidence_threshold, frame_width, frame_height

    # You can extend this endpoint to accept query parameters for adjustments
    # For now, just returning current settings
    return jsonify({
        'confidence_threshold': confidence_threshold,
        'resolution': f"{frame_width}x{frame_height}",
        'coordinate_precision': 6,
        'tracking_tolerance': tracking_tolerance
    })

if __name__ == '__main__':
    print("🌐 Starting Enhanced High-Precision YOLOv8 Flask App...")
    print("🎯 Features: 6-decimal precision, tracking, enhanced confidence scoring")
    print(f"📺 Resolution: {frame_width}x{frame_height} for maximum precision")
    print("📱 View at: http://localhost:5000")
    print("📊 Enhanced API Endpoints:")
    print("   - /api/crowd_coordinates - High-precision coordinates")
    print("   - /api/area_analytics_precise - Detailed area analytics")
    print("   - /api/adjust_precision - Precision settings")
    app.run(debug=True, use_reloader=False, host='0.0.0.0', port=5000, threaded=True)
