import cv2
import yaml
import logging
import time
import numpy as np
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, List

# Components
from src.ingestion.stream_loader import StreamLoader
from src.reconstruction.inpainter import FaceInpainter
from src.recognition.tracker import IOUTracker
from database.vector_db import VectorDB
from dashboard.app import start_dashboard, update_frame, update_logs

# InsightFace
from insightface.app import FaceAnalysis

# Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("Main")

import os
from dotenv import load_dotenv

# Load .env file
load_dotenv()

def load_config(path: str = "settings.yaml") -> Dict:
    with open(path, 'r') as f:
        content = f.read()
        
    # Expand environment variables in the YAML content
    # matches ${VAR} or $VAR
    expanded_content = os.path.expandvars(content)
    
    return yaml.safe_load(expanded_content)

import requests
import json
import serial
import pynmea2

def get_physical_gps(config):
    """Attempt to read from physical GPS sensor."""
    gps_conf = config.get('system', {}).get('gps_device', {})
    if not gps_conf.get('enabled', False):
        return None
        
    port = gps_conf.get('port', 'COM3')
    baud = gps_conf.get('baudrate', 9600)
    
    logger.info(f"Attempting to connect to GPS at {port}...")
    try:
        with serial.Serial(port, baud, timeout=2) as ser:
            # Read a few lines to find a valid fix
            for _ in range(20):
                line = ser.readline().decode('utf-8', errors='ignore')
                if line.startswith('$GPGGA') or line.startswith('$GPRMC'):
                    msg = pynmea2.parse(line)
                    if hasattr(msg, 'latitude') and msg.latitude != 0:
                        logger.info(f"Physical GPS Fix: {msg.latitude}, {msg.longitude}")
                        return {
                            'lat': msg.latitude,
                            'lng': msg.longitude,
                            'city': 'GPS Fixed'
                        }
    except Exception as e:
        logger.warning(f"Physical GPS failed: {e}")
        
    return None

import asyncio

async def get_windows_gps():
    """Access Windows Location API via WinSDK."""
    try:
        from winsdk.windows.devices.geolocation import Geolocator, GeolocationAccessStatus
        
        # Request access explicitly
        logger.info("Requesting Windows Location Access...")
        status = await Geolocator.request_access_async()
        
        if status == GeolocationAccessStatus.ALLOWED:
            logger.info("Access Granted. Acquiring signal...")
            locator = Geolocator()
            locator.desired_accuracy_in_meters = 10
            
            # Timeout after 5 seconds
            timeout = 5000 
            # In UWP, getting position is simpler, but in python winsdk proper async await is needed
            # We use a lambda to map the timeout/optimization
            pos = await locator.get_geoposition_async()
            
            lat = pos.coordinate.point.position.latitude
            lng = pos.coordinate.point.position.longitude
            logger.info(f"Windows GPS Fix: {lat}, {lng}")
            return {'lat': lat, 'lng': lng, 'city': 'Windows GPS'}
            
        elif status == GeolocationAccessStatus.DENIED:
            logger.warning("Windows Location Access DENIED.")
            logger.warning("ACTION REQUIRED: Go to Settings > Privacy > Location and turn on 'Allow desktop apps to access your location'.")
            
        else:
            logger.warning(f"Windows Location Access Status: {status}")
            
    except ImportError:
        logger.warning("winsdk not installed. Skipping Windows GPS.")
    except Exception as e:
        logger.warning(f"Windows GPS Error: {e}")
        
    return None

def get_live_location(config):
    """Detect location with Physical GPS -> Windows GPS -> IP Failover."""
    
    # 1. Try Physical Serial GPS
    gps = get_physical_gps(config)
    if gps:
        return gps
        
    # 2. Try Windows Inbuilt GPS
    try:
        # Run async function synchronously
        gps = asyncio.run(get_windows_gps())
        if gps:
            return gps
    except Exception as e:
        logger.warning(f"Async execution failed: {e}")

    # 3. IP-based Fallback
    providers = [
        ("http://ip-api.com/json", lambda d: (d['lat'], d['lon'], d.get('city', 'Unknown'))),
        ("https://ipinfo.io/json", lambda d: (float(d['loc'].split(',')[0]), float(d['loc'].split(',')[1]), d.get('city', 'Unknown'))),
        ("https://ipapi.co/json/", lambda d: (float(d['latitude']), float(d['longitude']), d.get('city', 'Unknown')))
    ]

    for url, parser in providers:
        try:
            logger.info(f"Detecting location via {url}...")
            response = requests.get(url, timeout=3)
            if response.status_code == 200:
                data = response.json()
                lat, lng, city = parser(data)
                logger.info(f"Location detected: {city} ({lat}, {lng})")
                return {'lat': lat, 'lng': lng, 'city': city}
        except Exception as e:
            logger.warning(f"Location fetch failed for {url}: {e}")
            
    return None

def send_to_hq(url, data):
    """Send detection data to HQ Server asynchronously."""
    if not url:
        return
    
    try:
        # Append /api/ingest if not present
        if not url.endswith('/api/ingest'):
            url = url.rstrip('/') + '/api/ingest'
            
        requests.post(url, json=data, timeout=2)
    except Exception as e:
        # Log only warnings to avoid spamming console on network issues
        logger.debug(f"Failed to send to HQ: {e}")

def main():
    config = load_config()
    
    # 0. Auto-Detect Location (if enabled or default)
    # Ideally we'd have a config flag, but user requested "automatically detect"
    detected_gps = get_live_location(config)
    
    if detected_gps:
        # Update system-wide GPS
        config.setdefault('system', {})
        config['system']['gps'] = detected_gps
        
        # Update Camera Defaults (if they don't have override or are local)
        # We assume local cameras (source 0/1) move with the device
        for cam in config.get('cameras', []):
            if isinstance(cam.get('source'), int): # Local webcam
                 cam['gps'] = detected_gps
    
    # Dashboard
    start_dashboard(config=config.get('system', {}), port=5000)

    # Database
    db = VectorDB(
        uri=config['database']['uri'], 
        collection_name=config['database']['collection_name'],
        dim=config['database']['dim']
    )
    
    # 2. Inpainting & Recognition Models
    logger.info("Loading models...")
    inpainter = FaceInpainter(
        model_path=config['inpainting']['model_path'],
        mask_threshold=config['inpainting']['mask_threshold']
    )
    
    app = FaceAnalysis(
        name=config['detection']['model_name'],
        providers=['CUDAExecutionProvider', 'CPUExecutionProvider'],
        allowed_modules=['detection', 'recognition', 'landmark_3d_68']
    )
    app.prepare(ctx_id=config['detection']['ctx_id'], det_thresh=config['detection']['det_thresh'])
    
    # 3. Stream & Tracker Initialization (Multi-Camera)
    cameras_config = config.get('cameras', [])
    streams = {}
    trackers = {}
    
    # Fallback if no cameras defined but stream key exists (legacy)
    if not cameras_config and 'stream' in config:
        cameras_config = [{
            'id': 'cam_default',
            'name': 'Default Camera',
            'source': config['stream']['source'],
            'active': True
        }]

    for cam in cameras_config:
        if not cam.get('active', True):
            continue
            
        cid = cam['id']
        logger.info(f"Initializing {cam['name']} ({cid})...")
        
        # Loader
        loader = StreamLoader(
            source=cam['source'],
            queue_size=config.get('stream', {}).get('queue_size', 5),
            width=config.get('stream', {}).get('width', 1280),
            height=config.get('stream', {}).get('height', 720)
        )
        loader.start()
        streams[cid] = {'loader': loader, 'config': cam}
        
        # Tracker
        trackers[cid] = {'tracker': IOUTracker(), 'identities': {}}

    if not streams:
        logger.error("No active cameras found!")
        return

    # Thread Pool
    executor = ThreadPoolExecutor(max_workers=4) # Increased for multi-cam
    # State
    fps_counter = 0
    start_time = time.time()
    
    # Persistent buffer for grid stitching (cid -> processed_frame)
    # Initialize with black frames
    grid_buffer = {} 
    last_frame_time = {}
    
    for cid, data in streams.items():
        blank = np.zeros((480, 640, 3), dtype=np.uint8)
        cv2.putText(blank, "Initializing...", (50, 240), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        grid_buffer[cid] = blank
        last_frame_time[cid] = time.time()

    try:
        while True:
            # Process each camera
            any_new_frame = False
            
            for cid, data in streams.items():
                loader = data['loader']
                cam_conf = data['config']
                
                # Non-blocking read
                ret, timestamp, frame = loader.read()
                
                if ret:
                    any_new_frame = True
                    last_frame_time[cid] = time.time()
                    
                    # --- Processing Pipeline ---
                    
                    # 1. Detection
                    try:
                        faces = app.get(frame)
                    except Exception as e:
                        logger.error(f"Detection error: {e}")
                        faces = []

                    # 2. Prepare for Tracker
                    detections = []
                    face_map = {}
                    for i, face in enumerate(faces):
                        bbox = face.bbox.astype(int)
                        detections.append([bbox[0], bbox[1], bbox[2], bbox[3]])
                        face_map[i] = face
                        
                    # 3. Update Camera-Specific Tracker
                    tracker_data = trackers[cid]
                    tracked_objects = tracker_data['tracker'].update(detections)
                    
                    for track_id, bbox in tracked_objects:
                        matched_face = None
                        for i, face in face_map.items():
                            f_bbox = face.bbox.astype(int)
                            if np.array_equal(f_bbox, bbox):
                                matched_face = face
                                break
                        
                        if matched_face is None: continue

                        # Identity
                        name = config['recognition']['unknown_label']
                        score = 0.0
                        local_identities = tracker_data['identities']
                        
                        if track_id in local_identities and local_identities[track_id]['score'] > 0.6:
                            name = local_identities[track_id]['name']
                            score = local_identities[track_id]['score']
                        else:
                            embedding = matched_face.embedding
                            matches = db.search_embedding(embedding, threshold=config['recognition']['similarity_threshold'])
                            if matches:
                                name = matches[0]['name']
                                score = matches[0]['score']
                                local_identities[track_id] = {'name': name, 'score': score}
                                
                            # Extract 3D Mesh for forensics (if available)
                            mesh_data = None
                            if hasattr(matched_face, 'landmark_3d_68') and matched_face.landmark_3d_68 is not None:
                                mesh_data = matched_face.landmark_3d_68.tolist()

                            # Log (with location info!)
                            update_logs(name, score, 
                                      location=cam_conf.get('name', cid),
                                      gps=cam_conf.get('gps'),
                                      mesh=mesh_data)

                            # Send to HQ (Async)
                            hq_url = config.get('system', {}).get('central_server')
                            if hq_url:
                                payload = {
                                    'name': name,
                                    'score': float(score),
                                    'timestamp': time.time(),
                                    'location': cam_conf.get('name', cid),
                                    'gps': cam_conf.get('gps'),
                                    'device_id': config.get('system', {}).get('device_id', 'Unknown'),
                                    'mesh': mesh_data
                                }
                                executor.submit(send_to_hq, hq_url, payload)

                        # Visualization
                        color = (0, 255, 0) if name != config['recognition']['unknown_label'] else (0, 0, 255)
                        cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color, 2)
                        cv2.putText(frame, f"{name} ({score:.2f})", (bbox[0], bbox[1]-10), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                        
                        # 3D
                        if hasattr(matched_face, 'landmark_3d_68') and matched_face.landmark_3d_68 is not None:
                            for pt in matched_face.landmark_3d_68.astype(int):
                                cv2.circle(frame, (pt[0], pt[1]), 1, (0, 255, 255), -1)

                    # Overlay Camera Name
                    cv2.putText(frame, cam_conf['name'], (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                    
                    # Update Buffer
                    grid_buffer[cid] = cv2.resize(frame, (640, 480))
                
                else:
                    # No new frame. Check timeout.
                    if time.time() - last_frame_time[cid] > 2.0:
                        # Timeout - Signal Lost
                        blank = np.zeros((480, 640, 3), dtype=np.uint8)
                        cv2.putText(blank, cam_conf['name'], (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                        cv2.putText(blank, "Source Input Lost", (50, 240), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
                        grid_buffer[cid] = blank

            # Stitch Buffer to Grid
            if not any_new_frame:
                time.sleep(0.01) # Avoid spinning if no cameras updated
                # But we still continue to push the dashboard so it doesn't freeze
            
            frames_to_stitch = list(grid_buffer.values())
            
            # Stitching (Simple logic: linear horizontal, wrapping if > 2)
            # For 2 cams: HConcat. For 4: 2x2.
            count = len(frames_to_stitch)
            if count == 1:
                final_view = frames_to_stitch[0]
            elif count == 2:
                final_view = np.hstack(frames_to_stitch)
            else:
                # Basic 2-column layout
                rows = []
                for i in range(0, count, 2):
                    chunk = frames_to_stitch[i:i+2]
                    if len(chunk) == 1: # Pad with black if odd
                        chunk.append(np.zeros((480, 640, 3), dtype=np.uint8))
                    rows.append(np.hstack(chunk))
                final_view = np.vstack(rows)

            # 5. Update and FPS
            if final_view is not None and final_view.size > 0:
                update_frame(final_view)
            else:
                logger.warning("Main Loop: Final view is empty or None")

            fps_counter += 1
            if time.time() - start_time > 1.0:
                logger.info(f"FPS: {fps_counter}")
                fps_counter = 0
                start_time = time.time()

    except KeyboardInterrupt:
        logger.info("Stopping...")
    finally:
        for data in streams.values():
            data['loader'].stop()
        executor.shutdown()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
