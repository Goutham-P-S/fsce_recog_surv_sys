from flask import Flask, Response, render_template, jsonify
import cv2
import threading
import logging
import time
import json
import os
import requests
import numpy as np
from collections import deque

logger = logging.getLogger("Dashboard")
app = Flask(__name__)

# Shared buffer
frame_buffer = None
buffer_lock = threading.Lock()

# Logs buffer (store last 50 logs)
logs_buffer = deque(maxlen=50)
logs_lock = threading.Lock()

# Analytics state
HISTORY_FILE = "detection_history.json"
analytics_data = {
    "detections_over_time": {}, # "HH:MM": count
    "top_criminals": {}, # "Name": count
    "total_detections": 0
}
analytics_lock = threading.Lock()
system_config = {}

def load_history():
    if os.path.exists(HISTORY_FILE):
        try:
            with open(HISTORY_FILE, 'r') as f:
                return json.load(f)
        except:
            return []
    return []

def save_history_entry(entry):
    """Append a single entry to history file efficiently."""
    try:
        with open(HISTORY_FILE, 'a') as f:
            f.write(json.dumps(entry) + "\n")
    except Exception as e:
        logger.error(f"Failed to save history: {e}")

def update_frame(frame):
    """Update the frame to be served."""
    global frame_buffer
    with buffer_lock:
        frame_buffer = frame.copy()

def update_logs(name, score, location="Unknown", gps=None, mesh=None):
    """Add a new log entry and update analytics."""
    timestamp = time.time()
    
    # 1. Update Live Log Buffer
    with logs_lock:
        logs_buffer.appendleft({
            "timestamp": timestamp,
            "name": name,
            "score": float(score),
            "location": location
            # We don't send mesh to live buffer to keep websocket light
        })
    
    # 2. Update Analytics & Persistence
    with analytics_lock:
        analytics_data["total_detections"] += 1
        
        # Confirmed match? Save to history
        if name != "Unknown":
            analytics_data["top_criminals"][name] = analytics_data["top_criminals"].get(name, 0) + 1
            
            # Use specific camera GPS or fallback to system GPS
            final_gps = gps if gps else system_config.get("gps", {"lat": 0, "lng": 0})

            # Persist for tracking
            entry = {
                "timestamp": timestamp,
                "name": name,
                "score": float(score),
                "location": location,
                "device_id": system_config.get("device_id", "Unknown"),
                "gps": final_gps,
                "mesh": mesh # Store 3D structural data for forensics
            }
            save_history_entry(entry)
            
            # --- DISTRIBUTED SYNC ---
            hq_url = system_config.get("central_server")
            if hq_url:
                def push_to_hq(payload, url):
                    try:
                        requests.post(f"{url}/api/ingest", json=payload, timeout=2)
                    except:
                        pass
                
                threading.Thread(target=push_to_hq, args=(entry, hq_url)).start()

        # Time bucket (minute resolution for demo)
        t_str = time.strftime("%H:%M")
        analytics_data["detections_over_time"][t_str] = analytics_data["detections_over_time"].get(t_str, 0) + 1

@app.route("/api/trail/<name>")
def get_trail(name):
    """Get the movement history of a specific person."""
    trail = []
    if os.path.exists(HISTORY_FILE):
        try:
            with open(HISTORY_FILE, 'r') as f:
                for line in f:
                    try:
                        entry = json.loads(line)
                        if entry['name'] == name:
                            trail.append(entry)
                    except:
                        pass
        except Exception:
            pass
            
    # Sort by time
    trail.sort(key=lambda x: x['timestamp'])
    return jsonify(trail)

def generate():
    """Video streaming generator function."""
    while True:
        try:
            with buffer_lock:
                if frame_buffer is None:
                    # Create a waiting placeholder
                    placeholder = np.zeros((480, 640, 3), dtype=np.uint8)
                    cv2.putText(placeholder, "Waiting for Camera...", (180, 240), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
                    (flag, encodedImage) = cv2.imencode(".jpg", placeholder)
                else:
                    # Encode frame
                    (flag, encodedImage) = cv2.imencode(".jpg", frame_buffer)
                
                if not flag:
                    continue
            
            # Yield the output frame in the byte format
            yield(b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + bytearray(encodedImage) + b'\r\n')
            
            time.sleep(0.04) # Limit to ~25fps to save bandwidth
        except Exception as e:
            logger.error(f"Stream generation error: {e}")
            time.sleep(0.1)

@app.route("/")
def index():
    return render_template("index.html")

@app.route('/api/history/<name>')
def get_history(name):
    """Get movement history for a specific person."""
    try:
        track = []
        if os.path.exists(HISTORY_FILE):
            with open(HISTORY_FILE, 'r') as f:
                for line in f:
                    try:
                        entry = json.loads(line)
                        if entry.get('name') == name:
                            gps = entry.get('gps')
                            if gps and isinstance(gps, dict) and 'lat' in gps and 'lng' in gps:
                                if gps['lat'] != 0 or gps['lng'] != 0: # Include if either lat or lng is non-zero
                                     track.append({
                                         'lat': gps['lat'],
                                         'lng': gps['lng'],
                                         'timestamp': entry.get('timestamp')
                                     })
                    except json.JSONDecodeError:
                        logger.warning(f"Skipping malformed JSON line in history file: {line.strip()}")
                    except Exception as e:
                        logger.error(f"Error processing history entry: {e}")
        
        # Sort by time
        track.sort(key=lambda x: x['timestamp'])
        return jsonify(track)
    except Exception as e:
        logger.error(f"Error in get_history for {name}: {e}")
        return jsonify({"error": str(e)}), 500

@app.route("/video_feed")
def video_feed():
    return Response(generate(), mimetype="multipart/x-mixed-replace; boundary=frame")

@app.route("/api/logs")
def get_logs():
    with logs_lock:
        return jsonify(list(logs_buffer))

@app.route("/api/analytics")
def get_analytics():
    with analytics_lock:
        # Sort top criminals
        sorted_criminals = dict(sorted(analytics_data["top_criminals"].items(), key=lambda x: x[1], reverse=True)[:5])
        
        return jsonify({
            "system": system_config,
            "stats": {
                "total": analytics_data["total_detections"],
                "top_criminals": sorted_criminals,
                "timeline": analytics_data["detections_over_time"]
            }
        })

def start_dashboard(config=None, host='0.0.0.0', port=5000):
    """Run the Flask app in a thread."""
    global system_config
    if config:
        system_config = config

    def run():
        # Disable banner
        cli = logging.getLogger('werkzeug')
        cli.setLevel(logging.ERROR)
        app.run(host=host, port=port, debug=False, use_reloader=False, threaded=True)
    
    t = threading.Thread(target=run, daemon=True)
    t.start()
    logger.info(f"Dashboard started at http://{host}:{port}")
