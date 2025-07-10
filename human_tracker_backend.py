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
    """
    The core backend class for the Human Tracking System. This class handles
    human detection, object tracking, facial recognition, low-light enhancement,
    and persistence of known faces. It is designed to be independent of the GUI
    and can be integrated into various frontends.
    """
    def __init__(self):
        """
        Initializes the HumanTrackerBackend. Sets up the YOLO model, defines
        face recognition thresholds, configures data persistence paths, and
        loads any previously saved known faces.
        """
        # --- Configuration ---
        # Determine the appropriate device for YOLO model (CUDA, MPS, or CPU)
        self.device = 'cpu'
        if torch.cuda.is_available():
            self.device = 'cuda'
        elif torch.backends.mps.is_available():
            self.device = 'mps'
        print(f"Using device for YOLO: {self.device}")
        # Load the YOLOv8 nano model and move it to the selected device
        self.yolo_model = YOLO('yolov8n.pt').to(self.device)

        # Face recognition thresholds and limits
        self.FACE_RECOGNITION_DISTANCE_THRESHOLD = 0.6  # Lower value means stricter match
        self.MAX_FACE_IMAGES_PER_PERSON = 15            # Max number of face encodings to store per person
        self.MIN_DISTINCT_FACE_DISTANCE = 0.4           # Min distance for a new encoding to be considered distinct

        # Persistence path for storing known faces database
        self.data_dir = os.path.join(os.path.expanduser("~"), ".yolo_human_tracker")
        os.makedirs(self.data_dir, exist_ok=True)  # Create directory if it doesn't exist
        self.KNOWN_FACES_DB_PATH = os.path.join(self.data_dir, "known_faces.pkl")

        # Tracking variables
        self.tracked_objects = {}  # Stores currently tracked human objects
        self.next_person_id_counter = 0  # Counter for assigning new person IDs
        self.known_faces_db = {}   # Database of known faces (name: {encodings, image, name})

        self.load_known_faces()  # Load known faces from disk on startup

        # State variables for toggling features
        self.face_recognition_enabled = True
        self.low_light_enhancement_enabled = False
        self.paused = False  # Controls whether frame processing is paused

        # --- Drawing Configuration ---
        self.BOX_COLORS = {
            "recognized": (0, 255, 0),      # Green for recognized
            "unidentified": (0, 165, 255),  # Orange for unidentified
            "unknown": (0, 0, 255),         # Red for unknown/no face
            "face": (255, 0, 0)             # Blue for face bounding box
        }
        self.FONT = cv2.FONT_HERSHEY_SIMPLEX
        self.FONT_SCALE = 0.6
        self.FONT_THICKNESS = 2


    # --- Persistence Functions ---
    def load_known_faces(self):
        """
        Loads the known faces database from a pickle file. If the file does not
        exist or an error occurs during loading, it initializes an empty database.
        It also handles backward compatibility for older data structures.
        """
        if os.path.exists(self.KNOWN_FACES_DB_PATH):
            try:
                with open(self.KNOWN_FACES_DB_PATH, 'rb') as f:
                    loaded_data = pickle.load(f)
                    self.known_faces_db = loaded_data.get('known_faces_db', {}) # Get the main dict
                    
                    # Find the highest 'Unidentified_' ID to continue numbering from there
                    max_unidentified_id = -1
                    for face_id in self.known_faces_db.keys():
                        if isinstance(face_id, str) and face_id.startswith("Unidentified_"):
                            try:
                                num_part = int(face_id.split('_')[1])
                                max_unidentified_id = max(max_unidentified_id, num_part)
                            except (ValueError, IndexError):
                                pass # Ignore malformed Unidentified IDs
                    self.next_person_id_counter = max_unidentified_id + 1

                    # Ensure data integrity and backward compatibility for each entry
                    for face_id, data in self.known_faces_db.items():
                        if 'image' not in data: data['image'] = None
                        if 'name' not in data: data['name'] = 'Unknown'
                        # Convert encodings to deque if not already, for fixed-size collection
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
        """
        Saves the current known faces database to a pickle file. This ensures
        that learned face data persists across application sessions.
        """
        data = {
            'known_faces_db': self.known_faces_db,
        }
        try:
            with open(self.KNOWN_FACES_DB_PATH, 'wb') as f:
                pickle.dump(data, f) # Serialize and save the database
            print(f"Saved {len(self.known_faces_db)} known faces to {self.KNOWN_FACES_DB_PATH}")
        except Exception as e:
            print(f"Error saving known faces database: {e}")

    # --- Feature Toggling ---
    def toggle_face_recognition(self):
        """
        Toggles the state of face recognition (enabled/disabled).
        Returns the new state.
        """
        self.face_recognition_enabled = not self.face_recognition_enabled
        return self.face_recognition_enabled

    def toggle_low_light_enhancement(self):
        """
        Toggles the state of low-light enhancement (enabled/disabled).
        Returns the new state.
        """
        self.low_light_enhancement_enabled = not self.low_light_enhancement_enabled
        return self.low_light_enhancement_enabled

    def process_frame(self, frame):
        """
        Main frame processing pipeline. This method takes a raw video frame,
        performs human detection, tracks individuals, and applies facial
        recognition if enabled. It updates the internal state of tracked objects
        and returns the processed frame along with tracking data.

        Args:
            frame (numpy.ndarray): The current video frame (BGR format).

        Returns:
            tuple: A tuple containing:
                - processed_frame (numpy.ndarray): The frame after processing (e.g., enhancement).
                - tracked_objects (dict): A dictionary of currently tracked objects.
                - known_faces_db (dict): The updated known faces database.
        """
        # Apply low-light enhancement if enabled
        if self.low_light_enhancement_enabled:
            # Convert to LAB color space for CLAHE (Contrast Limited Adaptive Histogram Equalization)
            lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
            cl = clahe.apply(l)
            limg = cv2.merge((cl, a, b))
            frame = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
            # Apply bilateral filter for noise reduction while preserving edges
            frame = cv2.bilateralFilter(frame, d=9, sigmaColor=75, sigmaSpace=75)

        # --- YOLO Human Detection ---
        # Run YOLO model on the current frame to detect human instances
        yolo_results = self.yolo_model(frame, device=self.device, verbose=False)
        current_human_detections = []
        for r in yolo_results:
            for box in r.boxes:
                # Filter detections to only include 'person' class
                if self.yolo_model.names[int(box.cls)] == 'person':
                    x1, y1, x2, y2 = map(int, box.xyxy[0]) # Bounding box coordinates
                    conf = float(box.conf[0]) # Confidence score
                    current_human_detections.append([x1, y1, x2, y2, conf])

        # --- Object Tracking and Association ---
        # Calculate centroids for current human detections
        current_human_centroids = [(int((x1 + x2) / 2), int((y1 + y2) / 2)) for x1, y1, x2, y2, _ in current_human_detections]
        
        # Build a list of all known face encodings for matching against new detections
        known_face_encodings_list = []
        known_face_ids_list = []
        for face_id, data in self.known_faces_db.items():
            if data.get('encodings'):
                known_face_encodings_list.extend(list(data['encodings']))
                known_face_ids_list.extend([face_id] * len(data['encodings']))

        # --- Core Tracking Logic ---
        next_tracked_objects = {} # Dictionary to store tracked objects for the next frame
        used_detection_indices = set() # Keep track of detections already associated

        # 1. Match existing tracked objects with current detections
        # This loop tries to find a corresponding detection for each person tracked in the previous frame.
        for person_id, obj_data in self.tracked_objects.items():
            min_dist = float('inf')
            best_match_idx = -1
            for i, centroid in enumerate(current_human_centroids):
                if i in used_detection_indices: # Skip detections already used
                    continue
                dist = np.linalg.norm(np.array(obj_data['centroid']) - np.array(centroid))
                if dist < 150:  # Distance threshold for association
                    if dist < min_dist:
                        min_dist = dist
                        best_match_idx = i
            
            if best_match_idx != -1:
                det = current_human_detections[best_match_idx]
                # Carry over existing data (like face_id, name) and update box/centroid
                next_tracked_objects[person_id] = {
                    **obj_data,
                    'box': det[:4],
                    'centroid': current_human_centroids[best_match_idx],
                }
                used_detection_indices.add(best_match_idx) # Mark detection as used

        # 2. Add new detections as new tracked objects
        # This loop processes detections that were not matched with existing tracked objects.
        for i in range(len(current_human_detections)):
            if i in used_detection_indices:
                continue # Skip if this detection was already used for an existing track
            det = current_human_detections[i]
            # Assign a new temporary person ID
            new_person_id = f"person_{self.next_person_id_counter}"
            self.next_person_id_counter += 1
            next_tracked_objects[new_person_id] = {
                'box': det[:4],
                'centroid': current_human_centroids[i],
                'face_id': None, # Initialize face_id as None for new persons
                'face_rect': None,
                'face_confidence': 0.0,
                'name': 'Unknown'
            }

        # 3. Process faces for all tracked objects
        # This loop iterates through all currently tracked objects to perform face recognition.
        for person_id, obj_data in next_tracked_objects.items():
            current_face_id = obj_data.get('face_id')
            # Check if the person has already been identified (either by name or as Unidentified_X)
            is_identified = current_face_id is not None and current_face_id != "No Face Detected"

            # Only perform face recognition if enabled and the person is not yet identified
            if self.face_recognition_enabled and not is_identified:
                hx1, hy1, hx2, hy2 = obj_data['box']
                human_roi = frame[hy1:hy2, hx1:hx2] # Extract the human region of interest

                if human_roi.shape[0] > 0 and human_roi.shape[1] > 0:
                    rgb_human_roi = cv2.cvtColor(human_roi, cv2.COLOR_BGR2RGB) # Convert to RGB for face_recognition
                    face_locations = face_recognition.face_locations(rgb_human_roi) # Detect faces within the ROI

                    if face_locations: # If a face is detected
                        face_rect = face_locations[0] # Take the first detected face
                        obj_data['face_rect'] = face_rect # Store face bounding box relative to ROI
                        try:
                            encoding = face_recognition.face_encodings(rgb_human_roi, [face_rect])[0] # Get face encoding
                            match_found = False
                            if known_face_encodings_list: # Check against known faces if any exist
                                distances = face_recognition.face_distance(known_face_encodings_list, encoding)
                                best_match_idx = np.argmin(distances)
                                if distances[best_match_idx] < self.FACE_RECOGNITION_DISTANCE_THRESHOLD: # If a match is found
                                    match_found = True
                                    face_id = known_face_ids_list[best_match_idx]
                                    obj_data.update({
                                        'face_id': face_id,
                                        'name': self.known_faces_db[face_id].get('name', 'Unknown'),
                                        'face_confidence': 1.0 - distances[best_match_idx]
                                    })
                                    print(f"Recognized {obj_data['name']}.")
                            
                            if not match_found: # If no match found, create a new Unidentified person
                                new_face_id = f"Unidentified_{self.next_person_id_counter}"
                                self.next_person_id_counter += 1
                                top, right, bottom, left = face_rect
                                face_img = human_roi[top:bottom, left:right] # Extract face image
                                self.known_faces_db[new_face_id] = {
                                    'encodings': deque([encoding], maxlen=self.MAX_FACE_IMAGES_PER_PERSON),
                                    'image': cv2.resize(face_img, (100, 100)) if face_img.size > 0 else None,
                                    'name': 'Unknown'
                                } # Store encoding and image
                                obj_data['face_id'] = new_face_id
                                print(f"New unidentified person saved: {new_face_id}")

                        except IndexError:
                            # Handle cases where encoding fails (e.g., face too small)
                            obj_data['face_id'] = "No Face Detected"
                    else:
                        # No face detected in the human ROI
                        obj_data['face_id'] = "No Face Detected"

        self.tracked_objects = next_tracked_objects # Update the main tracked objects dictionary
        
        # Draw overlays on the frame
        display_frame = self.draw_overlays(frame.copy(), self.tracked_objects)

        return display_frame, self.tracked_objects, self.known_faces_db

    def draw_overlays(self, frame, tracked_objects):
        """
        Draws bounding boxes, labels, and other information on the frame.

        Args:
            frame (numpy.ndarray): The frame to draw on.
            tracked_objects (dict): The dictionary of tracked objects.

        Returns:
            numpy.ndarray: The frame with overlays.
        """
        for person_id, obj_data in tracked_objects.items():
            x1, y1, x2, y2 = obj_data['box']
            face_id_label = obj_data.get('face_id', 'Unknown')
            face_confidence = obj_data.get('face_confidence', 0.0)
            
            # Determine the person's name for display
            person_name = self.known_faces_db.get(face_id_label, {}).get('name', str(face_id_label))
            
            label = f"P:{person_id} ({person_name})"
            
            # Assign colors based on identification status
            if isinstance(face_id_label, str) and not face_id_label.startswith("Unidentified_") and face_id_label != "No Face Detected" and person_name != str(face_id_label):
                label += f" C:{face_confidence:.2f}"
                color = self.BOX_COLORS["recognized"]
            elif face_id_label.startswith("Unidentified_"):
                label = f"P:{person_id} (Unidentified)"
                color = self.BOX_COLORS["unidentified"]
            else:
                color = self.BOX_COLORS["unknown"]

            # Draw bounding box and text label
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, self.FONT_THICKNESS)
            cv2.putText(frame, label, (x1, y1 - 10), self.FONT, self.FONT_SCALE, color, self.FONT_THICKNESS)

            # Draw face bounding box if available
            face_rect = obj_data.get('face_rect')
            if face_rect:
                top, right, bottom, left = face_rect
                fx1_abs, fy1_abs = x1 + left, y1 + top
                fx2_abs, fy2_abs = x1 + right, y1 + bottom
                cv2.rectangle(frame, (fx1_abs, fy1_abs), (fx2_abs, fy2_abs), self.BOX_COLORS["face"], self.FONT_THICKNESS)  # Blue for face
        return frame

    def get_recognized_person_data(self):
        """
        Retrieves data for the most confidently recognized person in the current frame.
        This is used to update the GUI's info panel.
        """
        recognized_faces_in_frame = []
        for p_id, obj_data in self.tracked_objects.items():
            # Filter for identified faces with sufficient confidence
            if isinstance(obj_data.get('face_id'), str) and \
               not obj_data['face_id'].startswith("Unidentified_") and \
               obj_data['face_id'] != "No Face Detected" and \
               obj_data.get('face_confidence', 0.0) > self.FACE_RECOGNITION_DISTANCE_THRESHOLD:
                recognized_faces_in_frame.append(obj_data)
        
        if recognized_faces_in_frame:
            # Sort by confidence to get the most confident recognition
            recognized_faces_in_frame.sort(key=lambda x: x.get('face_confidence', 0.0), reverse=True)
            return recognized_faces_in_frame[0]
        return None # No recognized person in frame

    def name_unidentified_person(self, target_face_id, new_name):
        """
        Renames an 'Unidentified' person to a proper name. This involves moving
        their data in the `known_faces_db` and updating any active tracked objects.

        Args:
            target_face_id (str): The 'Unidentified_X' ID of the person to rename.
            new_name (str): The new name to assign to the person.

        Returns:
            tuple: (bool, str) - True if successful, False otherwise, along with a message.
        """
        # Basic validation for the new name
        if not new_name or new_name.isspace():
            return False, "New name cannot be empty."

        # Prevent naming conflicts
        if new_name in self.known_faces_db:
            return False, f"The name '{new_name}' already exists in the database."

        if target_face_id in self.known_faces_db: # Ensure the target ID exists
            unidentified_person_data = self.known_faces_db[target_face_id] # Get data

            # Create a new entry with the new name as the key
            self.known_faces_db[new_name] = {
                'encodings': unidentified_person_data.get('encodings', deque(maxlen=self.MAX_FACE_IMAGES_PER_PERSON)),
                'image': unidentified_person_data.get('image'),
                'name': new_name  # Set the name explicitly
            }

            del self.known_faces_db[target_face_id] # Remove the old 'Unidentified' entry

            # Update any currently tracked objects that were using the old ID
            for person_id, obj_data in self.tracked_objects.items():
                if obj_data.get('face_id') == target_face_id:
                    self.tracked_objects[person_id]['face_id'] = new_name
                    self.tracked_objects[person_id]['name'] = new_name
            
            self.save_known_faces() # Save changes to disk
            return True, f"Successfully renamed '{target_face_id}' to '{new_name}'."
            
        return False, f"Could not find '{target_face_id}' to name."

    def get_unidentified_face_ids(self):
        """
        Returns a list of 'Unidentified' face IDs currently being tracked.
        This is used by the GUI for the naming feature.
        """
        unidentified_face_ids = []
        for p_id, obj_data in self.tracked_objects.items():
            if isinstance(obj_data.get('face_id'), str) and obj_data['face_id'].startswith("Unidentified_"):
                unidentified_face_ids.append(obj_data['face_id'])
        return unidentified_face_ids

    def get_known_person_names(self):
        """
        Returns a sorted list of all people with a proper name (not 'Unidentified_X' or 'Unknown').
        This is used by the GUI for the 'Edit Name' feature.
        """
        known_names = []
        for face_id, data in self.known_faces_db.items():
            if not face_id.startswith("Unidentified_") and data.get("name", "Unknown") != "Unknown":
                known_names.append(data["name"])
        return sorted(list(set(known_names))) # Return a sorted list of unique names

    def edit_person_name(self, old_name, new_name):
        """
        Edits the name of an already recognized person. This involves updating
        their key in the `known_faces_db` and any active tracked objects.

        Args:
            old_name (str): The current name of the person.
            new_name (str): The new name to assign.

        Returns:
            tuple: (bool, str) - True if successful, False otherwise, along with a message.
        """
        # Basic validation for the new name
        if not new_name or new_name.isspace():
            return False, "New name cannot be empty."

        # Prevent naming conflicts
        if new_name in self.known_faces_db:
            return False, f"The name '{new_name}' already exists."

        if old_name in self.known_faces_db: # Ensure the old name exists
            # Move the data from the old key to the new key
            self.known_faces_db[new_name] = self.known_faces_db.pop(old_name)
            self.known_faces_db[new_name]['name'] = new_name # Update the name within the data

            # Update any currently tracked objects that were using the old name
            for person_id, obj_data in self.tracked_objects.items():
                if obj_data.get('face_id') == old_name:
                    self.tracked_objects[person_id]['face_id'] = new_name
                    self.tracked_objects[person_id]['name'] = new_name
            
            self.save_known_faces() # Save changes to disk
            return True, f"Successfully renamed '{old_name}' to '{new_name}'."
        
        return False, f"Could not find '{old_name}' to rename."