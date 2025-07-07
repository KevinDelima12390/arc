import cv2
import time
import platform
from human_tracker_backend import HumanTrackerBackend

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

# Initialize the backend
backend = HumanTrackerBackend()

paused = False
zoom_enabled = False
prev_time = 0
last_recognized_person_data = None

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
    elif key == ord('n'):
        unidentified_face_ids = backend.get_unidentified_face_ids()
        if unidentified_face_ids:
            target_face_id = unidentified_face_ids[0] # Take the first one for simplicity
            print(f"\n--- Naming Person (Face ID: {target_face_id}) ---")
            new_name = input("Enter a name for this person (or leave blank to cancel): ")
            if new_name:
                if backend.name_unidentified_person(target_face_id, new_name):
                    print(f"Assigned name '{new_name}' to face ID {target_face_id}.")
                else:
                    print(f"Error: Face ID {target_face_id} not found in database.")
            else:
                print("Naming cancelled.")
        else:
            print("No 'Unidentified' person currently tracked to name.")
    elif key == ord('w'):
        fr_status = backend.toggle_face_recognition()
        print(f"Facial Recognition: {'Enabled' if fr_status else 'Disabled'}")
    elif key == ord('l'):
        ll_status = backend.toggle_low_light_enhancement()
        print(f"Low-Light Enhancement: {'Enabled' if ll_status else 'Disabled'}")

    if not paused:
        success, frame = cap.read()
        if not success:
            print("Error: Failed to read frame from camera.")
            break

        # Process frame using the backend
        processed_frame, tracked_objects, known_faces_db = backend.process_frame(frame.copy())

        # Calculate FPS
        curr_time = time.time()
        fps = 1 / (curr_time - prev_time) if prev_time > 0 else 0
        prev_time = curr_time

        # --- Drawing and Display (simplified for now, will be moved to GUI) ---
        # This part will be replaced by the PyQt6 GUI
        display_frame = processed_frame.copy()

        # Draw bounding boxes and labels
        for person_id, obj_data in tracked_objects.items():
            x1, y1, x2, y2 = obj_data['box']
            face_id_label = obj_data.get('face_id', 'Unknown')
            face_confidence = obj_data.get('face_confidence', 0.0)
            person_name = known_faces_db.get(face_id_label, {}).get('name', str(face_id_label))
            label = f"Person {person_id} ({person_name})"
            if isinstance(face_id_label, str) and not face_id_label.startswith("Unidentified_") and face_id_label != "No Face Detected":
                label += f" C:{face_confidence:.2f}"

            color = (0, 255, 0)
            cv2.rectangle(display_frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(display_frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

            face_rect = obj_data.get('face_rect')
            if face_rect:
                top, right, bottom, left = face_rect
                fx1_abs, fy1_abs = x1 + left, y1 + top
                fx2_abs, fy2_abs = x1 + right, y1 + bottom
                cv2.rectangle(display_frame, (fx1_abs, fy1_abs), (fx2_abs, fy2_abs), (255, 0, 0), 2)

        # Display FPS
        cv2.putText(display_frame, f"FPS: {int(fps)}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        cv2.imshow("Human Tracking with Face Recognition", display_frame)
    else:
        if 'display_frame' in locals():
            paused_frame = display_frame.copy()
            cv2.putText(paused_frame, "Paused", (paused_frame.shape[1] // 2 - 50, paused_frame.shape[0] // 2), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
            cv2.imshow("Human Tracking with Face Recognition", paused_frame)
        else:
            print("Paused. Press 'p' to unpause.")
            time.sleep(0.1)

# Save faces before exiting
backend.save_known_faces()

cap.release()
cv2.destroyAllWindows()
