import cv2
import numpy as np
from ultralytics import YOLO
import torch
import time
import os
import pickle
from collections import deque
import face_recognition
import platform

# --- Configuration ---
# OS Detection
print(f"Operating System: {platform.system()}")

# YOLOv8 Model
device = 'cpu'
if torch.cuda.is_available():
    device = 'cuda'
elif torch.backends.mps.is_available():
    device = 'mps'
print(f"Using device for YOLO: {device}")
yolo_model = YOLO('yolov8n.pt').to(device)

# Face Detection Model (OpenCV DNN Caffe) - Still used for initial face localization
# NOTE: face_recognition library handles face detection and alignment internally
# for encoding, so the OpenCV DNN detector is not strictly needed for the core FR part,
# but it's kept here from potentially older versions or for alternative detection methods.
# For this script using face_recognition, the DNN part is redundant for the FR flow
# but could be used as an *alternative* detector if face_recognition.face_locations was slow.
# We will rely solely on face_recognition.face_locations for simplicity here.
# script_dir = os.path.dirname(__file__)
# face_detector_prototxt = os.path.join(script_dir, "deploy.prototxt")
# face_detector_caffemodel = os.path.join(script_dir, "res10_300x300_ssd_iter_140000_fp16.caffemodel")
# if os.path.exists(face_detector_prototxt) and os.path.exists(face_detector_caffemodel):
#     face_detector = cv2.dnn.readNetFromCaffe(face_detector_prototxt, face_detector_caffemodel)
# else:
#     print("Warning: Face detector prototxt or caffemodel not found. Face detection might be slower.")
#     face_detector = None # Or handle loading failure

# Thresholds
# FACE_DETECTION_CONF_THRESHOLD = 0.7 # Not directly used with face_recognition.face_locations
FACE_RECOGNITION_DISTANCE_THRESHOLD = 0.6 # Lower is more strict for face_recognition library
MAX_FACE_IMAGES_PER_PERSON = 15
MIN_DISTINCT_FACE_DISTANCE = 0.4

# Persistence path
data_dir = os.path.join(os.path.expanduser("~"), ".yolo_human_tracker")
os.makedirs(data_dir, exist_ok=True)
KNOWN_FACES_DB_PATH = os.path.join(data_dir, "known_faces.pkl")

# Video capture
def get_camera_backend():
    os_name = platform.system()
    if os_name == "Windows":
        return cv2.CAP_DSHOW  # DirectShow for better performance on Windows
    elif os_name == "Darwin":
        return cv2.CAP_AVFOUNDATION # AVFoundation for macOS
    else:
        return cv2.CAP_ANY # Default for Linux and others

cap = cv2.VideoCapture(0, get_camera_backend())
if not cap.isOpened():
    print("Error: Could not open video stream.")
    exit()


# Tracking variables
tracked_objects = {} # {person_id: {'box': [x1,y1,x2,y2], 'centroid': (cx, cy), 'face_id': str, 'face_rect': tuple, 'face_confidence': float, 'name': str}}
next_person_id_counter = 0 # Use a different variable name to avoid confusion with assigned face_id

# Face recognition database
# {face_id: {'encodings': deque([face_encoding1, face_encoding2, ...]), 'image': face_image, 'name': str}}
known_faces_db = {}

# --- Persistence Functions ---
def load_known_faces():
    global known_faces_db, next_person_id_counter
    if os.path.exists(KNOWN_FACES_DB_PATH):
        try:
            with open(KNOWN_FACES_DB_PATH, 'rb') as f:
                loaded_data = pickle.load(f)
                known_faces_db = loaded_data.get('known_faces_db', {})
                # Update next_person_id_counter based on existing 'Unidentified_' IDs
                max_unidentified_id = -1
                for face_id in known_faces_db.keys():
                    if isinstance(face_id, str) and face_id.startswith("Unidentified_"):
                        try:
                            num_part = int(face_id.split('_')[1])
                            max_unidentified_id = max(max_unidentified_id, num_part)
                        except (ValueError, IndexError):
                            pass # Ignore malformed IDs
                next_person_id_counter = max_unidentified_id + 1

                # Ensure all entries have the 'image' and 'name' keys, for backward compatibility
                for face_id, data in known_faces_db.items():
                    if 'image' not in data:
                        data['image'] = None # Initialize with None
                    if 'name' not in data:
                        data['name'] = 'Unknown' # Initialize with 'Unknown'
                    # Ensure encodings is a deque
                    if not isinstance(data['encodings'], deque):
                         data['encodings'] = deque(data['encodings'], maxlen=MAX_FACE_IMAGES_PER_PERSON)

            print(f"Loaded {len(known_faces_db)} known faces from {KNOWN_FACES_DB_PATH}")
        except Exception as e:
            print(f"Error loading known faces database: {e}. Starting fresh.")
            known_faces_db = {}
            next_person_id_counter = 0
    else:
        print("No known faces database found. Starting fresh.")

def save_known_faces():
    data = {
        'known_faces_db': known_faces_db,
        # next_person_id_counter is implicitly handled as new unidentified IDs are created
    }
    try:
        with open(KNOWN_FACES_DB_PATH, 'wb') as f:
            pickle.dump(data, f)
        print(f"Saved {len(known_faces_db)} known faces to {KNOWN_FACES_DB_PATH}")
    except Exception as e:
        print(f"Error saving known faces database: {e}")


# Load faces at startup
load_known_faces()

# Pause and FPS variables
paused = False
zoom_enabled = False # New: Toggle zoom state
prev_time = 0
frame_count = 0
# last_known_human_boxes = [] # This variable is not used
last_recognized_person_data = None # Store data for the person displayed in the side panel
face_recognition_enabled = True # Initialize toggle state

# --- Main Loop ---
while True:
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('p'):
        paused = not paused
    elif key == ord('z'):
        zoom_enabled = not zoom_enabled
        print(f"Zoom: {'Enabled' if zoom_enabled else 'Disabled'}")
    elif key == ord('n'): # New: Naming feature
        # Find tracked persons with "Unidentified" face_id
        unidentified_tracked_person_ids = [
            p_id for p_id, obj_data in tracked_objects.items()
            if isinstance(obj_data.get('face_id'), str) and obj_data['face_id'].startswith("Unidentified_")
        ]

        if unidentified_tracked_person_ids:
            # For simplicity, let's prompt to name the first one found
            target_person_id = unidentified_tracked_person_ids[0]
            face_id_to_name = tracked_objects[target_person_id]['face_id']

            print(f"\n--- Naming Person {target_person_id} (Face ID: {face_id_to_name}) ---")
            new_name = input("Enter a name for this person (or leave blank to cancel): ")
            if new_name:
                # Update the name in known_faces_db using the face_id
                if face_id_to_name in known_faces_db:
                    known_faces_db[face_id_to_name]['name'] = new_name
                    print(f"Assigned name '{new_name}' to face ID {face_id_to_name}.")
                    # Update the tracked object's name immediately
                    tracked_objects[target_person_id]['name'] = new_name
                    save_known_faces() # Save changes immediately
                else:
                    print(f"Error: Face ID {face_id_to_name} not found in database.")
            else:
                print("Naming cancelled.")
        else:
            print("No 'Unidentified' person currently tracked to name.")
    elif key == ord('w'): # New: Toggle facial recognition
        face_recognition_enabled = not face_recognition_enabled
        print(f"Facial Recognition: {'Enabled' if face_recognition_enabled else 'Disabled'}")


    if not paused:
        success, frame = cap.read()
        if not success:
            print("Error: Failed to read frame from camera.")
            break

        # Calculate FPS
        curr_time = time.time()
        fps = 1 / (curr_time - prev_time) if prev_time > 0 else 0
        prev_time = curr_time

        # --- YOLO Human Detection ---
        # YOLO results format: r.boxes has attributes like xyxy, conf, cls
        yolo_results = yolo_model(frame, device=device, verbose=False)
        current_human_detections = [] # List of [x1, y1, x2, y2, confidence]
        for r in yolo_results:
            for box in r.boxes:
                if yolo_model.names[int(box.cls)] == 'person':
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    conf = float(box.conf[0])
                    current_human_detections.append([x1, y1, x2, y2, conf])

        # --- Object Tracking and Association ---
        current_human_centroids = [(int((x1 + x2) / 2), int((y1 + y2) / 2)) for x1, y1, x2, y2, _ in current_human_detections]

        next_tracked_objects = {}
        used_current_indices = set()

        # 1. Match existing tracked objects to current detections
        for person_id, obj_data in tracked_objects.items():
            min_dist = float('inf')
            best_match_idx = -1

            for i, centroid in enumerate(current_human_centroids):
                if i in used_current_indices:
                    continue # Skip detections already matched

                dist = np.linalg.norm(np.array(obj_data['centroid']) - np.array(centroid))
                if dist < 100 and dist < min_dist: # Max distance threshold for matching
                    min_dist = dist
                    best_match_idx = i

            if best_match_idx != -1:
                # Found a match
                det = current_human_detections[best_match_idx]
                next_tracked_objects[person_id] = {
                    'box': det[:4],
                    'centroid': current_human_centroids[best_match_idx],
                    'face_id': obj_data.get('face_id', None), # Preserve existing face_id
                    'face_rect': obj_data.get('face_rect', None), # Preserve face_rect
                    'face_confidence': obj_data.get('face_confidence', 0.0), # Preserve confidence
                    'name': obj_data.get('name', 'Unknown') # Preserve name
                }
                used_current_indices.add(best_match_idx)

        # 2. Process unmatched current detections (these are potentially new persons)
        for i in range(len(current_human_detections)):
            if i not in used_current_indices:
                det = current_human_detections[i]
                # Assign a temporary ID based on the detection index for this frame
                temp_id = f"temp_{i}_{int(time.time())}" # Use a temporary ID until face ID is assigned
                next_tracked_objects[temp_id] = {
                    'box': det[:4],
                    'centroid': current_human_centroids[i],
                    'face_id': None, # Face ID needs to be determined
                    'face_rect': None,
                    'face_confidence': 0.0,
                    'name': 'Unknown'
                }

        # 3. Process Faces for all tracked objects in next_tracked_objects
        # Create lists of known encodings and their corresponding face_ids for faster comparison
        known_face_encodings_list = []
        known_face_ids_list = []
        # Exclude 'No Face Detected' string from known_faces_db keys
        for face_id, data in known_faces_db.items():
             # Only add known individuals or existing unidentified ones
             if isinstance(face_id, str) and not face_id.startswith("Unidentified_") and face_id != "No Face Detected":
                 known_face_encodings_list.extend(list(data['encodings']))
                 known_face_ids_list.extend([face_id] * len(data['encodings']))
             # Also include current 'Unidentified_' IDs from the database for matching
             elif isinstance(face_id, str) and face_id.startswith("Unidentified_"):
                 known_face_encodings_list.extend(list(data['encodings']))
                 known_face_ids_list.extend([face_id] * len(data['encodings']))


        final_tracked_objects = {} # Build the final list after face processing

        # Store face data to add/update in known_faces_db after iterating
        face_updates_to_save = {} # {face_id: {'encoding': encoding, 'image': image, 'name': name}}

        for person_id, obj_data in list(next_tracked_objects.items()): # Iterate on a copy as we might change person_id
            hx1, hy1, hx2, hy2 = obj_data['box']
            human_roi = frame[hy1:hy2, hx1:hx2]

            current_face_id = obj_data.get('face_id') # Get the face_id status from the tracker
            assigned_face_id = current_face_id # Start with the current status
            face_rect_in_roi = obj_data.get('face_rect') # Start with existing face_rect
            face_confidence = obj_data.get('face_confidence', 0.0)
            person_name = obj_data.get('name', 'Unknown')

            # Only attempt face recognition if enabled and face_id is None or Unidentified
            # Also, if face_id is "No Face Detected", re-attempt detection
            needs_face_processing = face_recognition_enabled and \
                                    (current_face_id is None or \
                                     (isinstance(current_face_id, str) and current_face_id.startswith("Unidentified_")) or \
                                     current_face_id == "No Face Detected")

            detected_face_encoding = None
            new_face_rect_in_roi = None # Store the newly detected face location

            if needs_face_processing and human_roi.shape[0] > 0 and human_roi.shape[1] > 0:
                rgb_human_roi = cv2.cvtColor(human_roi, cv2.COLOR_BGR2RGB)
                face_locations_in_roi = face_recognition.face_locations(rgb_human_roi)

                if face_locations_in_roi:
                    # Assuming one face per human ROI for simplicity, take the best quality one?
                    # For now, just take the first one found.
                    new_face_rect_in_roi = face_locations_in_roi[0] # (top, right, bottom, left)
                    try:
                         detected_face_encoding = face_recognition.face_encodings(rgb_human_roi, [new_face_rect_in_roi])[0]
                    except IndexError:
                         # This can happen if face_locations finds something but face_encodings fails (rare)
                         print("Warning: Could not get encoding for detected face.")
                         detected_face_encoding = None


            if detected_face_encoding is not None:
                # Attempt to match the detected face encoding
                if known_face_encodings_list:
                    face_distances = face_recognition.face_distance(known_face_encodings_list, detected_face_encoding)
                    best_match_index = np.argmin(face_distances)
                    min_distance = face_distances[best_match_index]

                    if min_distance < FACE_RECOGNITION_DISTANCE_THRESHOLD:
                        # Matched a known face
                        assigned_face_id = known_face_ids_list[best_match_index]
                        face_confidence = 1.0 - min_distance
                        person_name = known_faces_db.get(assigned_face_id, {}).get('name', 'Unknown')

                        # Add new encoding to the known person's database if distinct and not full
                        if len(known_faces_db[assigned_face_id]['encodings']) < MAX_FACE_IMAGES_PER_PERSON:
                            # Check for distinctness against *existing* encodings for *this specific person*
                            is_distinct = True
                            # Get only encodings for the matched ID
                            target_encodings = list(known_faces_db[assigned_face_id]['encodings'])
                            if target_encodings: # Avoid comparing against empty list
                                distances_to_person = face_recognition.face_distance(target_encodings, detected_face_encoding)
                                if np.min(distances_to_person) < MIN_DISTINCT_FACE_DISTANCE:
                                    is_distinct = False

                            if is_distinct:
                                # Mark for saving later to avoid modifying db during iteration
                                face_updates_to_save[assigned_face_id] = {
                                    'encoding': detected_face_encoding,
                                    'image': None, # Will capture image below if needed
                                    'name': person_name # Preserve name
                                }
                                print(f"Added new distinct encoding for {assigned_face_id}.")
                                # Capture and store the face image if adding a new encoding
                                top, right, bottom, left = new_face_rect_in_roi
                                face_img = rgb_human_roi[top:bottom, left:right]
                                if face_img.shape[0] > 0 and face_img.shape[1] > 0:
                                     face_updates_to_save[assigned_face_id]['image'] = cv2.resize(face_img, (100, 100)) # Standardize size


                    else:
                        # Not a match, treat as a new unidentified face
                        assigned_face_id = f"Unidentified_{next_person_id_counter}"
                        face_confidence = 0.0 # Unidentified confidence is 0
                        person_name = 'Unknown'

                        # Check if this specific "Unidentified_N" ID already exists in the database (shouldn't happen with counter logic, but belt-and-suspenders)
                        while assigned_face_id in known_faces_db:
                             next_person_id_counter += 1
                             assigned_face_id = f"Unidentified_{next_person_id_counter}"

                        next_person_id_counter += 1 # Increment counter for the next new unidentified face

                        # Mark for saving as a new unidentified person
                        face_updates_to_save[assigned_face_id] = {
                            'encodings': deque([detected_face_encoding], maxlen=MAX_FACE_IMAGES_PER_PERSON),
                            'image': None, # Will capture image below
                            'name': 'Unknown' # Name is Unknown initially
                        }
                        # Capture and store the face image
                        top, right, bottom, left = new_face_rect_in_roi
                        face_img = rgb_human_roi[top:bottom, left:right]
                        if face_img.shape[0] > 0 and face_img.shape[1] > 0:
                            face_updates_to_save[assigned_face_id]['image'] = cv2.resize(face_img, (100, 100)) # Standardize size
                        print(f"New unidentified face detected, assigned ID: {assigned_face_id}")

                else:
                    # No known faces in DB yet, treat as the first new unidentified face
                    assigned_face_id = f"Unidentified_{next_person_id_counter}"
                    face_confidence = 0.0
                    person_name = 'Unknown'
                    next_person_id_counter += 1

                    # Mark for saving as the first unidentified person
                    face_updates_to_save[assigned_face_id] = {
                         'encodings': deque([detected_face_encoding], maxlen=MAX_FACE_IMAGES_PER_PERSON),
                         'image': None, # Will capture image below
                         'name': 'Unknown'
                    }
                    # Capture and store the face image
                    top, right, bottom, left = new_face_rect_in_roi
                    face_img = rgb_human_roi[top:bottom, left:right]
                    if face_img.shape[0] > 0 and face_img.shape[1] > 0:
                         face_updates_to_save[assigned_face_id]['image'] = cv2.resize(face_img, (100, 100)) # Standardize size
                    print(f"First face detected, assigned ID: {assigned_face_id}")

            elif needs_face_processing:
                # Face recognition is enabled, but no face was detected in the ROI
                assigned_face_id = "No Face Detected" # Indicate that face detection failed
                face_rect_in_roi = None
                face_confidence = 0.0
                person_name = 'Unknown' # Default name if no face is found

            # Update object data for the final list
            final_tracked_objects[person_id] = { # Keep the original person_id from matching stage
                'box': obj_data['box'],
                'centroid': obj_data['centroid'],
                'face_id': assigned_face_id,
                'face_rect': new_face_rect_in_roi if new_face_rect_in_roi is not None else face_rect_in_roi, # Use new if found, else keep old
                'face_confidence': face_confidence,
                'name': person_name # Use the determined name
            }

        # Apply updates to known_faces_db after the iteration
        for face_id, update_data in face_updates_to_save.items():
            if face_id.startswith("Unidentified_") and 'encodings' in update_data:
                # This is a new unidentified person
                known_faces_db[face_id] = update_data # Add the new entry
            elif face_id in known_faces_db:
                # This is an existing person (known or unidentified being updated)
                if 'encoding' in update_data:
                    known_faces_db[face_id]['encodings'].append(update_data['encoding']) # Add the new encoding
                if update_data.get('image') is not None:
                    known_faces_db[face_id]['image'] = update_data['image'] # Update the image
                if update_data.get('name') != 'Unknown': # Only update name if a specific name was set
                     known_faces_db[face_id]['name'] = update_data['name'] # Update name


        # Replace tracked_objects with the processed list
        tracked_objects = final_tracked_objects

        # --- Drawing and Display ---
        # Create a larger canvas for video + side panel
        panel_width = 250 # Width of the side panel
        display_width = frame.shape[1] + panel_width
        display_height = frame.shape[0]

        full_display_frame = np.zeros((display_height, display_width, 3), dtype=np.uint8)

        # Create a mutable copy of the video frame for drawing
        # This will be the left side of the full display
        video_output_frame = frame.copy()

        # Side panel background
        cv2.rectangle(full_display_frame, (frame.shape[1], 0), (display_width, display_height), (50, 50, 50), -1)

        # Update last_recognized_person_data for side panel display
        recognized_faces_in_frame = []
        for p_id, obj_data in tracked_objects.items():
            # Consider a person recognized if face_id is not Unidentified or No Face Detected
            # and confidence is above a display threshold (optional, but good practice)
            if isinstance(obj_data.get('face_id'), str) and \
               not obj_data['face_id'].startswith("Unidentified_") and \
               obj_data['face_id'] != "No Face Detected" and \
               obj_data.get('face_confidence', 0.0) > FACE_RECOGNITION_DISTANCE_THRESHOLD: # Use the matching threshold for display too
                recognized_faces_in_frame.append(obj_data)

        # Update the person shown in the side panel
        if recognized_faces_in_frame:
            # Sort by confidence and pick the highest
            recognized_faces_in_frame.sort(key=lambda x: x.get('face_confidence', 0.0), reverse=True)
            last_recognized_person_data = recognized_faces_in_frame[0]
        # If no one recognized in this frame, keep displaying the last recognized person data
        # else: pass # Don't clear last_recognized_person_data if no one is recognized

        # Display recognized person in side panel
        if last_recognized_person_data and \
           isinstance(last_recognized_person_data.get('face_id'), str) and \
           not last_recognized_person_data['face_id'].startswith("Unidentified_") and \
           last_recognized_person_data['face_id'] != "No Face Detected": # Only display truly recognized people
            face_id = last_recognized_person_data['face_id']
            person_name = known_faces_db.get(face_id, {}).get('name', 'Unknown') # Get name from DB
            confidence = last_recognized_person_data['face_confidence']
            ref_image = known_faces_db.get(face_id, {}).get('image')

            text_x = frame.shape[1] + 10
            text_y = 30

            cv2.putText(full_display_frame, "Recognized:", (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
            text_y += 25
            cv2.putText(full_display_frame, f"Name: {person_name}", (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
            text_y += 25
            cv2.putText(full_display_frame, f"ID: {face_id}", (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
            text_y += 25
            cv2.putText(full_display_frame, f"Conf: {confidence:.2f}", (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
            text_y += 25

            if ref_image is not None:
                img_h, img_w, _ = ref_image.shape
                img_x = frame.shape[1] + (panel_width - img_w) // 2
                img_y = text_y + 10
                # Ensure image fits within panel
                if img_y + img_h < display_height:
                    full_display_frame[img_y:img_y+img_h, img_x:img_x+img_w] = ref_image
                else:
                    # Optionally resize image if too large, or just skip
                    print("Warning: Reference image too large for side panel, skipping.")
        else:
            # Display message when no recognized person is currently shown
            text_x = frame.shape[1] + 10
            text_y = display_height // 2
            cv2.putText(full_display_frame, "No recognized", (text_x, text_y - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
            cv2.putText(full_display_frame, "person", (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
            cv2.putText(full_display_frame, "detected", (text_x, text_y + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)


        # --- Draw bounding boxes and labels on video_output_frame ---
        # This loop draws the green human box for EVERY tracked object.
        # It also draws the blue face box if available.
        
        if zoom_enabled and len(tracked_objects) == 1:
            # Automatic Digital Zoom on the single tracked person
            person_id = list(tracked_objects.keys())[0]
            x1, y1, x2, y2 = tracked_objects[person_id]['box']

            # Add padding to the human ROI for zoom
            padding = 100 # Pixels
            zoom_x1 = max(0, x1 - padding)
            zoom_y1 = max(0, y1 - padding)
            zoom_x2 = min(frame.shape[1], x2 + padding)
            zoom_y2 = min(frame.shape[0], y2 + padding)

            zoomed_roi = frame[zoom_y1:zoom_y2, zoom_x1:zoom_x2]

            if zoomed_roi.shape[0] > 0 and zoomed_roi.shape[1] > 0:
                # Resize zoomed_roi to original frame size
                video_output_frame = cv2.resize(zoomed_roi, (frame.shape[1], frame.shape[0]))

                # Adjust coordinates for drawing on the zoomed frame
                scale_x = frame.shape[1] / zoomed_roi.shape[1]
                scale_y = frame.shape[0] / zoomed_roi.shape[0]

                # Redraw the single tracked object on the zoomed frame
                # Note: This loop still iterates through ALL tracked_objects, but we only expect 1 here
                for p_id, obj_data in tracked_objects.items():
                    hx1, hy1, hx2, hy2 = obj_data['box']

                    # Adjust human box coordinates
                    hx1_zoom = int((hx1 - zoom_x1) * scale_x)
                    hy1_zoom = int((hy1 - zoom_y1) * scale_y)
                    hx2_zoom = int((hx2 - zoom_x1) * scale_x)
                    hy2_zoom = int((hy2 - zoom_y1) * scale_y)

                    face_id_label = obj_data.get('face_id', 'Unknown')
                    face_confidence = obj_data.get('face_confidence', 0.0)
                    person_name = known_faces_db.get(face_id_label, {}).get('name', str(face_id_label)) # Get name from DB, fallback to face_id_label string
                    label = f"Person {p_id} ({person_name})"
                    if isinstance(face_id_label, str) and not face_id_label.startswith("Unidentified_") and face_id_label != "No Face Detected":
                         label += f" C:{face_confidence:.2f}"

                    color = (0, 255, 0) # Green for human box
                    # Draw the green human bounding box on the zoomed frame
                    cv2.rectangle(video_output_frame, (hx1_zoom, hy1_zoom), (hx2_zoom, hy2_zoom), color, 2)
                    cv2.putText(video_output_frame, label, (hx1_zoom, hy1_zoom - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

                    # Adjust face box coordinates if available (face_rect is relative to human_roi)
                    face_rect = obj_data.get('face_rect') # (top, right, bottom, left) relative to human_roi
                    if face_rect:
                        top, right, bottom, left = face_rect
                        # Convert face_rect from human_roi coords to original frame coords
                        fx1_abs_orig = hx1 + left
                        fy1_abs_orig = hy1 + top
                        fx2_abs_orig = hx1 + right
                        fy2_abs_orig = hy1 + bottom

                        # Adjust face box coordinates to the zoomed frame
                        fx1_zoom = int((fx1_abs_orig - zoom_x1) * scale_x)
                        fy1_zoom = int((fy1_abs_orig - zoom_y1) * scale_y)
                        fx2_zoom = int((fx2_abs_orig - zoom_x1) * scale_x)
                        fy2_zoom = int((fy2_abs_orig - zoom_y1) * scale_y)
                        # Draw the blue face bounding box on the zoomed frame
                        cv2.rectangle(video_output_frame, (fx1_zoom, fy1_zoom), (fx2_zoom, fy2_zoom), (255, 0, 0), 2) # Blue for face box

                        # Display stored reference image if available and recognized (on main video)
                        if isinstance(face_id_label, str) and not face_id_label.startswith("Unidentified_") and face_id_label != "No Face Detected" and face_id_label in known_faces_db and known_faces_db[face_id_label].get('image') is not None:
                             ref_image = known_faces_db[face_id_label]['image']
                             # Overlay the reference image near the face bounding box on the zoomed frame
                             img_h, img_w, _ = ref_image.shape

                             # Position the reference image to the right of the face box
                             overlay_x = fx2_zoom + 10
                             overlay_y = fy1_zoom

                             # Ensure it's within frame boundaries
                             if overlay_x + img_w > video_output_frame.shape[1]:
                                 overlay_x = fx1_zoom - img_w - 10 # Try left side if right is out of bounds
                             if overlay_x < 0: # If still out of bounds, place at 0
                                 overlay_x = 0

                             if overlay_y + img_h > video_output_frame.shape[0]:
                                 overlay_y = video_output_frame.shape[0] - img_h # Adjust if too low
                             if overlay_y < 0: # If still out of bounds, place at 0
                                 overlay_y = 0
                             
                             # Check bounds before slicing
                             if overlay_y >= 0 and overlay_y + img_h <= video_output_frame.shape[0] and \
                                overlay_x >= 0 and overlay_x + img_w <= video_output_frame.shape[1]:
                                 video_output_frame[overlay_y:overlay_y+img_h, overlay_x:overlay_x+img_w] = ref_image
                             else:
                                 print(f"Warning: Could not overlay ref image for {face_id_label} - outside zoomed frame bounds.")


            else:
                # If zoom ROI is invalid (e.g., human box too small), just show the original frame
                video_output_frame = frame.copy() # Revert to original if zoom failed
                zoom_enabled = False # Disable zoom if it failed
                print("Zoom ROI invalid, disabling zoom.")

        # --- Draw bounding boxes and labels on original frame (when not zoomed or zoom failed) ---
        if not zoom_enabled or video_output_frame.shape != frame.shape: # Check if zoom was applied successfully
             # Draw on original frame if not zooming or if zoom failed
            video_output_frame = frame.copy() # Ensure we are drawing on a fresh copy of the original frame

            for person_id, obj_data in tracked_objects.items():
                x1, y1, x2, y2 = obj_data['box']
                face_id_label = obj_data.get('face_id', 'Unknown')
                face_confidence = obj_data.get('face_confidence', 0.0)
                person_name = known_faces_db.get(face_id_label, {}).get('name', str(face_id_label)) # Get name from DB, fallback to face_id_label string
                label = f"Person {person_id} ({person_name})" # Use the tracker's person_id for the main label
                if isinstance(face_id_label, str) and not face_id_label.startswith("Unidentified_") and face_id_label != "No Face Detected":
                     label += f" C:{face_confidence:.2f}"

                color = (0, 255, 0) # Green for human box
                # --- Draw the green human bounding box for the tracked object ---
                cv2.rectangle(video_output_frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(video_output_frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

                face_rect = obj_data.get('face_rect') # (top, right, bottom, left) relative to human_roi
                if face_rect:
                    top, right, bottom, left = face_rect
                    # Convert face_rect from human_roi coords to original frame coords
                    fx1_abs, fy1_abs = x1 + left, y1 + top
                    fx2_abs, fy1_abs = x1 + right, y1 + top # fy1 is same
                    fx1_abs, fy2_abs = x1 + left, y1 + bottom # fx1 is same
                    fx2_abs, fy2_abs = x1 + right, y1 + bottom

                    # Ensure face box coordinates are within frame bounds before drawing
                    fx1_abs = max(0, fx1_abs)
                    fy1_abs = max(0, fy1_abs)
                    fx2_abs = min(video_output_frame.shape[1], fx2_abs)
                    fy2_abs = min(video_output_frame.shape[0], fy2_abs)

                    # Draw the blue face bounding box
                    if fx1_abs < fx2_abs and fy1_abs < fy2_abs: # Draw only if valid rectangle
                        cv2.rectangle(video_output_frame, (fx1_abs, fy1_abs), (fx2_abs, fy2_abs), (255, 0, 0), 2) # Blue for face box

                    # Display stored reference image if available and recognized (on main video)
                    if isinstance(face_id_label, str) and not face_id_label.startswith("Unidentified_") and face_id_label != "No Face Detected" and face_id_label in known_faces_db and known_faces_db[face_id_label].get('image') is not None:
                        ref_image = known_faces_db[face_id_label]['image']
                        # Overlay the reference image near the face bounding box
                        img_h, img_w, _ = ref_image.shape

                        # Position the reference image to the right of the face box
                        overlay_x = fx2_abs + 10
                        overlay_y = fy1_abs

                        # Ensure it's within frame boundaries
                        if overlay_x + img_w > video_output_frame.shape[1]:
                            overlay_x = fx1_abs - img_w - 10 # Try left side if right is out of bounds
                        if overlay_x < 0: # If still out of bounds, place at 0
                            overlay_x = 0

                        if overlay_y + img_h > video_output_frame.shape[0]:
                            overlay_y = video_output_frame.shape[0] - img_h # Adjust if too low
                        if overlay_y < 0: # If still out of bounds, place at 0
                            overlay_y = 0

                        # Check bounds before slicing
                        if overlay_y >= 0 and overlay_y + img_h <= video_output_frame.shape[0] and \
                           overlay_x >= 0 and overlay_x + img_w <= video_output_frame.shape[1]:
                            video_output_frame[overlay_y:overlay_y+img_h, overlay_x:overlay_x+img_w] = ref_image
                        else:
                             print(f"Warning: Could not overlay ref image for {face_id_label} - outside original frame bounds.")


        # Copy the potentially modified video_output_frame back to the left side of full_display_frame
        full_display_frame[0:display_height, 0:frame.shape[1]] = video_output_frame

        # Display FPS
        cv2.putText(full_display_frame, f"FPS: {int(fps)}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        # Display control info
        control_text_y = 60
        cv2.putText(full_display_frame, "'p' pause | 'q' quit", (10, control_text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 1)
        cv2.putText(full_display_frame, f"'z' zoom ({'On' if zoom_enabled else 'Off'}) | 'n' name", (10, control_text_y + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 1)
        cv2.putText(full_display_frame, f"'w' FR ({'On' if face_recognition_enabled else 'Off'})", (10, control_text_y + 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 1)


        cv2.imshow("Human Tracking with Face Recognition", full_display_frame)
    else:
        # If paused, keep displaying the last frame with a "Paused" message
        if 'full_display_frame' in locals(): # Check if full_display_frame has been created
            # Draw "Paused" on a copy to avoid redrawing it every frame
            paused_frame = full_display_frame.copy()
            cv2.putText(paused_frame, "Paused", (paused_frame.shape[1] // 2 - 50, paused_frame.shape[0] // 2), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
            cv2.imshow("Human Tracking with Face Recognition", paused_frame)
        else:
            # If paused before the first frame is read
            print("Paused. Press 'p' to unpause.")
            time.sleep(0.1) # Prevent high CPU usage while paused without a frame


# Save faces before exiting
save_known_faces()

cap.release()
cv2.destroyAllWindows()