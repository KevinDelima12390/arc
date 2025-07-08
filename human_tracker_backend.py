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

class HumanTrackerBackend:
    def __init__(self):
        # --- Configuration ---
        self.device = 'cpu'
        if torch.cuda.is_available():
            self.device = 'cuda'
        elif torch.backends.mps.is_available():
            self.device = 'mps'
        print(f"Using device for YOLO: {self.device}")
        self.yolo_model = YOLO('yolov8n.pt').to(self.device)

        self.FACE_RECOGNITION_DISTANCE_THRESHOLD = 0.6
        self.MAX_FACE_IMAGES_PER_PERSON = 15
        self.MIN_DISTINCT_FACE_DISTANCE = 0.4

        # Persistence path
        self.data_dir = os.path.join(os.path.expanduser("~"), ".yolo_human_tracker")
        os.makedirs(self.data_dir, exist_ok=True)
        self.KNOWN_FACES_DB_PATH = os.path.join(self.data_dir, "known_faces.pkl")

        # Tracking variables
        self.tracked_objects = {}
        self.next_person_id_counter = 0
        self.known_faces_db = {}

        self.load_known_faces()

        # State variables
        self.face_recognition_enabled = True
        self.low_light_enhancement_enabled = False
        self.paused = False # Add paused state to backend

    # --- Persistence Functions ---
    def load_known_faces(self):
        if os.path.exists(self.KNOWN_FACES_DB_PATH):
            try:
                with open(self.KNOWN_FACES_DB_PATH, 'rb') as f:
                    loaded_data = pickle.load(f)
                    self.known_faces_db = loaded_data.get('known_faces_db', {})
                    max_unidentified_id = -1
                    for face_id in self.known_faces_db.keys():
                        if isinstance(face_id, str) and face_id.startswith("Unidentified_"):
                            try:
                                num_part = int(face_id.split('_')[1])
                                max_unidentified_id = max(max_unidentified_id, num_part)
                            except (ValueError, IndexError):
                                pass
                    self.next_person_id_counter = max_unidentified_id + 1

                    for face_id, data in self.known_faces_db.items():
                        if 'image' not in data:
                            data['image'] = None
                        if 'name' not in data:
                            data['name'] = 'Unknown'
                        if not isinstance(data['encodings'], deque):
                            data['encodings'] = deque(data['encodings'], maxlen=self.MAX_FACE_IMAGES_PER_PERSON)

                print(f"Loaded {len(self.known_faces_db)} known faces from {self.KNOWN_FACES_DB_PATH}")
            except Exception as e:
                print(f"Error loading known faces database: {e}. Starting fresh.")
                self.known_faces_db = {}
                self.next_person_id_counter = 0
        else:
            print("No known faces database found. Starting fresh.")

    def save_known_faces(self):
        data = {
            'known_faces_db': self.known_faces_db,
        }
        try:
            with open(self.KNOWN_FACES_DB_PATH, 'wb') as f:
                pickle.dump(data, f)
            print(f"Saved {len(self.known_faces_db)} known faces to {self.KNOWN_FACES_DB_PATH}")
        except Exception as e:
            print(f"Error saving known faces database: {e}")

    def toggle_face_recognition(self):
        self.face_recognition_enabled = not self.face_recognition_enabled
        return self.face_recognition_enabled

    def toggle_low_light_enhancement(self):
        self.low_light_enhancement_enabled = not self.low_light_enhancement_enabled
        return self.low_light_enhancement_enabled

    def process_frame(self, frame):
        # Apply low-light enhancement if enabled
        if self.low_light_enhancement_enabled:
            lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
            cl = clahe.apply(l)
            limg = cv2.merge((cl, a, b))
            frame = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
            frame = cv2.bilateralFilter(frame, d=9, sigmaColor=75, sigmaSpace=75)

        # --- YOLO Human Detection ---
        yolo_results = self.yolo_model(frame, device=self.device, verbose=False)
        current_human_detections = []
        for r in yolo_results:
            for box in r.boxes:
                if self.yolo_model.names[int(box.cls)] == 'person':
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    conf = float(box.conf[0])
                    current_human_detections.append([x1, y1, x2, y2, conf])

        # --- Object Tracking and Association ---
        current_human_centroids = [(int((x1 + x2) / 2), int((y1 + y2) / 2)) for x1, y1, x2, y2, _ in current_human_detections]

        next_tracked_objects = {}
        used_current_indices = set()

        for person_id, obj_data in self.tracked_objects.items():
            min_dist = float('inf')
            best_match_idx = -1

            for i, centroid in enumerate(current_human_centroids):
                if i in used_current_indices:
                    continue

                dist = np.linalg.norm(np.array(obj_data['centroid']) - np.array(centroid))
                if dist < 100 and dist < min_dist:
                    min_dist = dist
                    best_match_idx = i

            if best_match_idx != -1:
                det = current_human_detections[best_match_idx]
                next_tracked_objects[person_id] = {
                    'box': det[:4],
                    'centroid': current_human_centroids[best_match_idx],
                    'face_id': obj_data.get('face_id', None),
                    'face_rect': obj_data.get('face_rect', None),
                    'face_confidence': obj_data.get('face_confidence', 0.0),
                    'name': obj_data.get('name', 'Unknown')
                }
                used_current_indices.add(best_match_idx)

        for i in range(len(current_human_detections)):
            if i not in used_current_indices:
                det = current_human_detections[i]
                temp_id = f"temp_{i}_{int(time.time())}"
                next_tracked_objects[temp_id] = {
                    'box': det[:4],
                    'centroid': current_human_centroids[i],
                    'face_id': None,
                    'face_rect': None,
                    'face_confidence': 0.0,
                    'name': 'Unknown'
                }

        # 3. Process Faces for all tracked objects
        known_face_encodings_list = []
        known_face_ids_list = []
        for face_id, data in self.known_faces_db.items():
             # We only want to match against faces that have a proper name or are unidentified
             if (isinstance(face_id, str) and data.get('name', 'Unknown') != 'Unknown') or face_id.startswith("Unidentified_"):
                 known_face_encodings_list.extend(list(data['encodings']))
                 known_face_ids_list.extend([face_id] * len(data['encodings']))

        final_tracked_objects = {}
        face_updates_to_save = {}

        for person_id, obj_data in list(next_tracked_objects.items()):
            hx1, hy1, hx2, hy2 = obj_data['box']
            human_roi = frame[hy1:hy2, hx1:hx2]

            current_face_id = obj_data.get('face_id')
            assigned_face_id = current_face_id
            face_rect_in_roi = obj_data.get('face_rect')
            face_confidence = obj_data.get('face_confidence', 0.0)
            person_name = obj_data.get('name', 'Unknown')

            needs_face_processing = self.face_recognition_enabled and \
                                    (current_face_id is None or current_face_id == "No Face Detected")

            detected_face_encoding = None
            new_face_rect_in_roi = None

            if needs_face_processing and human_roi.shape[0] > 0 and human_roi.shape[1] > 0:
                rgb_human_roi = cv2.cvtColor(human_roi, cv2.COLOR_BGR2RGB)
                face_locations_in_roi = face_recognition.face_locations(rgb_human_roi)

                if face_locations_in_roi:
                    new_face_rect_in_roi = face_locations_in_roi[0]
                    try:
                         detected_face_encoding = face_recognition.face_encodings(rgb_human_roi, [new_face_rect_in_roi])[0]
                    except IndexError:
                         print("Warning: Could not get encoding for detected face.")
                         detected_face_encoding = None

            if detected_face_encoding is not None:
                match_found = False
                if known_face_encodings_list:
                    face_distances = face_recognition.face_distance(known_face_encodings_list, detected_face_encoding)
                    best_match_index = np.argmin(face_distances)
                    min_distance = face_distances[best_match_index]

                    if min_distance < self.FACE_RECOGNITION_DISTANCE_THRESHOLD:
                        match_found = True
                        assigned_face_id = known_face_ids_list[best_match_index]
                        face_confidence = 1.0 - min_distance
                        person_name = self.known_faces_db[assigned_face_id].get('name', 'Unknown')

                        # Add new encoding to the matched person if it's distinct
                        target_encodings = list(self.known_faces_db[assigned_face_id]['encodings'])
                        if target_encodings:
                            distances_to_person = face_recognition.face_distance(target_encodings, detected_face_encoding)
                            if np.min(distances_to_person) > self.MIN_DISTINCT_FACE_DISTANCE:
                                if len(self.known_faces_db[assigned_face_id]['encodings']) < self.MAX_FACE_IMAGES_PER_PERSON:
                                    self.known_faces_db[assigned_face_id]['encodings'].append(detected_face_encoding)
                                    print(f"Added new distinct encoding for {assigned_face_id}.")
                        else: # First encoding for a person somehow
                             self.known_faces_db[assigned_face_id]['encodings'].append(detected_face_encoding)


                if not match_found:
                    # No match found, create a new Unidentified person
                    assigned_face_id = f"Unidentified_{self.next_person_id_counter}"
                    face_confidence = 0.0
                    person_name = 'Unknown'

                    while assigned_face_id in self.known_faces_db:
                        self.next_person_id_counter += 1
                        assigned_face_id = f"Unidentified_{self.next_person_id_counter}"
                    
                    self.next_person_id_counter += 1

                    self.known_faces_db[assigned_face_id] = {
                        'encodings': deque([detected_face_encoding], maxlen=self.MAX_FACE_IMAGES_PER_PERSON),
                        'image': None,
                        'name': 'Unknown'
                    }
                    
                    top, right, bottom, left = new_face_rect_in_roi
                    face_img = rgb_human_roi[top:bottom, left:right]
                    if face_img.shape[0] > 0 and face_img.shape[1] > 0:
                        bgr_face_img = cv2.cvtColor(cv2.resize(face_img, (100, 100)), cv2.COLOR_RGB2BGR)
                        self.known_faces_db[assigned_face_id]['image'] = bgr_face_img
                    
                    print(f"New unidentified face detected, assigned ID: {assigned_face_id}")

            elif needs_face_processing:
                # No face was detected in the ROI for a new person
                assigned_face_id = "No Face Detected"
                face_rect_in_roi = None
                face_confidence = 0.0
                person_name = 'Unknown'

            # Update the tracked object with the new face info
            final_tracked_objects[person_id] = {
                'box': obj_data['box'],
                'centroid': obj_data['centroid'],
                'face_id': assigned_face_id,
                'face_rect': new_face_rect_in_roi if new_face_rect_in_roi is not None else face_rect_in_roi,
                'face_confidence': face_confidence,
                'name': person_name
            }

        self.tracked_objects = final_tracked_objects
        return frame, self.tracked_objects, self.known_faces_db

    def get_recognized_person_data(self):
        recognized_faces_in_frame = []
        for p_id, obj_data in self.tracked_objects.items():
            # print(f"Checking person {p_id}: face_id={obj_data.get('face_id')}, confidence={obj_data.get('face_confidence', 0.0)}")
            if isinstance(obj_data.get('face_id'), str) and \
               not obj_data['face_id'].startswith("Unidentified_") and \
               obj_data['face_id'] != "No Face Detected" and \
               obj_data.get('face_confidence', 0.0) > self.FACE_RECOGNITION_DISTANCE_THRESHOLD:
                recognized_faces_in_frame.append(obj_data)
        
        if recognized_faces_in_frame:
            recognized_faces_in_frame.sort(key=lambda x: x.get('face_confidence', 0.0), reverse=True)
            # print(f"Returning recognized person: {recognized_faces_in_frame[0].get('name')}")
            return recognized_faces_in_frame[0]
        # print("No recognized person data to return.")
        return None

    def name_unidentified_person(self, target_face_id, new_name):
        # Check if the new name already exists to avoid conflicts
        if new_name in self.known_faces_db:
            print(f"Error: The name '{new_name}' already exists in the database.")
            return False

        if target_face_id in self.known_faces_db:
            # 1. Get the data from the old "Unidentified" entry
            unidentified_person_data = self.known_faces_db[target_face_id]

            # 2. Create a new entry with the new name as the key
            self.known_faces_db[new_name] = {
                'encodings': unidentified_person_data.get('encodings', deque(maxlen=self.MAX_FACE_IMAGES_PER_PERSON)),
                'image': unidentified_person_data.get('image'),
                'name': new_name  # Set the name explicitly
            }

            # 3. Remove the old "Unidentified" entry
            del self.known_faces_db[target_face_id]

            # 4. Update any currently tracked objects that have the old face_id
            for person_id, obj_data in self.tracked_objects.items():
                if obj_data.get('face_id') == target_face_id:
                    self.tracked_objects[person_id]['face_id'] = new_name
                    self.tracked_objects[person_id]['name'] = new_name
            
            # 5. Save the updated database to disk
            self.save_known_faces()
            print(f"Successfully renamed '{target_face_id}' to '{new_name}'.")
            return True
            
        print(f"Error: Could not find '{target_face_id}' to name.")
        return False

    def get_unidentified_face_ids(self):
        unidentified_face_ids = []
        for p_id, obj_data in self.tracked_objects.items():
            if isinstance(obj_data.get('face_id'), str) and obj_data['face_id'].startswith("Unidentified_"):
                unidentified_face_ids.append(obj_data['face_id'])
        return unidentified_face_ids
