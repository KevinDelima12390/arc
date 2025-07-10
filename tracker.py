import cv2
import time
import platform
from human_tracker_backend import HumanTrackerBackend

# This file (tracker.py) provides a simplified, non-GUI version of the human tracking system.
# It's primarily used for command-line execution, testing the core backend logic,
# and demonstrating basic video processing without the PyQt6 overhead.

# Video capture setup
def get_camera_backend():
    """
    Determines the appropriate OpenCV video capture backend based on the operating system.
    This helps in optimizing camera performance and compatibility across different platforms.
    - Windows: Uses DirectShow (cv2.CAP_DSHOW) for better performance.
    - macOS: Uses AVFoundation (cv2.CAP_AVFOUNDATION) for native camera access.
    - Others (Linux, etc.): Uses the default backend (cv2.CAP_ANY).
    """
    os_name = platform.system()
    if os_name == "Windows":
        return cv2.CAP_DSHOW  # DirectShow for better performance on Windows
    elif os_name == "Darwin":
        return cv2.CAP_AVFOUNDATION # AVFoundation for macOS
    else:
        return cv2.CAP_ANY # Default for Linux and others

# Initialize video capture from the default camera (index 0) using the determined backend.
cap = cv2.VideoCapture(0, get_camera_backend())
if not cap.isOpened():
    print("Error: Could not open video stream.")
    exit() # Exit if camera cannot be opened

# Initialize the HumanTrackerBackend, which contains the core logic for detection, tracking, and recognition.
backend = HumanTrackerBackend()

# State variables for the tracker's operation
pause_processing = False # Controls whether frame processing is paused
zoom_enabled = False     # Placeholder for a zoom feature (not fully implemented in this version)
prev_frame_time = 0      # Stores the timestamp of the previous frame for FPS calculation
last_recognized_person_data = None # Stores data of the last recognized person (not directly used in this display loop)

# --- Main Loop ---
# This loop continuously reads frames, processes them, and displays the results.
# It also handles keyboard inputs for basic controls.
while True:
    # Read keyboard input (waitKey(1) waits for 1ms and returns key pressed)
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break # Press 'q' to quit the application
    elif key == ord('p'):
        pause_processing = not pause_processing # Press 'p' to toggle pause/resume
    elif key == ord('z'):
        zoom_enabled = not zoom_enabled # Press 'z' to toggle zoom (placeholder)
        print(f"Zoom: {'Enabled' if zoom_enabled else 'Disabled'}")
    elif key == ord('n'):
        # Press 'n' to name an unidentified person (simplified for command-line)
        unidentified_face_ids = backend.get_unidentified_face_ids()
        if unidentified_face_ids:
            target_face_id = unidentified_face_ids[0] # For simplicity, names the first detected unidentified person
            print(f"\n--- Naming Person (Face ID: {target_face_id}) ---")
            new_name = input("Enter a name for this person (or leave blank to cancel): ")
            if new_name:
                # Call backend function to rename the person
                if backend.name_unidentified_person(target_face_id, new_name):
                    print(f"Assigned name '{new_name}' to face ID {target_face_id}.")
                else:
                    print(f"Error: Face ID {target_face_id} not found in database.")
            else:
                print("Naming cancelled.")
        else:
            print("No 'Unidentified' person currently tracked to name.")
    elif key == ord('w'):
        # Press 'w' to toggle facial recognition
        fr_status = backend.toggle_face_recognition()
        print(f"Facial Recognition: {'Enabled' if fr_status else 'Disabled'}")
    elif key == ord('l'):
        # Press 'l' to toggle low-light enhancement
        ll_status = backend.toggle_low_light_enhancement()
        print(f"Low-Light Enhancement: {'Enabled' if ll_status else 'Disabled'}")

    if not pause_processing:
        success, frame = cap.read() # Read a frame from the camera
        if not success:
            print("Error: Failed to read frame from camera.")
            break # Exit loop if frame cannot be read

        # Process the frame using the HumanTrackerBackend
        display_frame, tracked_objects, known_faces_db = backend.process_frame(frame.copy())

        # Calculate Frames Per Second (FPS)
        curr_time = time.time()
        fps = 1 / (curr_time - prev_frame_time) if prev_frame_time > 0 else 0
        prev_frame_time = curr_time

        # Display FPS on the top-left corner of the frame
        cv2.putText(display_frame, f"FPS: {int(fps)}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        cv2.imshow("Human Tracking with Face Recognition", display_frame) # Display the frame
    else:
        # If paused, display the last frame with a "Paused" overlay
        if 'display_frame' in locals():
            paused_frame = display_frame.copy()
            cv2.putText(paused_frame, "Paused", (paused_frame.shape[1] // 2 - 50, paused_frame.shape[0] // 2), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
            cv2.imshow("Human Tracking with Face Recognition", paused_frame)
        else:
            print("Paused. Press 'p' to unpause.")
            time.sleep(0.1) # Small delay when paused to reduce CPU usage

# --- Cleanup ---
# Save known faces database before exiting the application
backend.save_known_faces()

# Release the camera resource and destroy all OpenCV windows
cap.release()
cv2.destroyAllWindows()