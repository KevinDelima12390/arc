# AI Human Tracking System

This project is a real-time human detection and tracking system designed for robust and intelligent monitoring using a webcam. It leverages advanced computer vision techniques to identify, track, and recognize individuals, providing a comprehensive solution for various applications.

## Status

**Version 1.5 is stable and fully functional.** This version incorporates significant improvements in facial recognition, tracking stability, and user interaction.

## Core Functionality & Technologies

The system is built upon a modular architecture, separating the core tracking and recognition logic (backend) from the graphical user interface (GUI).

### 1. Human Detection and Tracking
-   **Technology:** Utilizes **YOLOv8** (You Only Look Once) for highly efficient and accurate real-time human detection. YOLOv8 is a state-of-the-art object detection model that can identify people within video frames.
-   **Methodology:** Once humans are detected, a custom **centroid tracker** is employed to assign unique IDs to each individual and maintain their tracking across consecutive frames. This ensures consistent identification of persons as they move within the camera's view.
-   **Hardware Acceleration:** Leverages **Apple MPS** (Metal Performance Shaders) for significant performance gains on compatible Apple Silicon Macs, ensuring smooth real-time processing.

### 2. Facial Recognition and Learning
-   **Technology:** Employs the `face_recognition` library, which is built upon `dlib`'s state-of-the-art face recognition capabilities.
-   **Initial Identification:** When a new person is detected, the system attempts to identify their face. If the face matches an existing entry in the `known_faces_db`, the person is identified by their name. If no match is found, a new "Unidentified" profile is created for them.
-   **Continuous Learning:** The system continuously learns and refines its knowledge of known individuals. When a recognized person is re-identified, their facial data is analyzed. If a new, distinct facial encoding or a better quality image is captured, it's added to their profile, enhancing future recognition accuracy.
-   **Persistence:** Known faces, including their names, facial encodings, and reference images, are securely saved to a local database (`known_faces.pkl`) using `pickle`. This ensures that learned information persists across application sessions.

### 3. User Interaction & Naming
-   **Naming Unidentified Persons:** Users can assign a name to any "Unidentified" person currently being tracked. Upon naming, the system converts the temporary "Unidentified" ID into a permanent, named entry in the database, ensuring consistent recognition.
-   **Editing Names:** An "Edit Name" feature allows users to change the name of any already recognized person in the database, providing flexibility in managing identities.

### 4. Graphical User Interface (GUI)
-   **Technology:** Developed using **PyQt6**, a powerful Python binding for the Qt cross-platform application framework, providing a rich and responsive desktop experience.
-   **Real-time Display:** The GUI displays the live video feed with bounding boxes and labels for tracked individuals, including their assigned IDs and names.
-   **Controls:** Provides intuitive controls for:
    -   **Start/Stop:** Begin or end the video stream.
    -   **Pause/Resume:** Temporarily halt or restart video processing.
    -   **Face Recognition Toggle:** Enable or disable facial recognition processing.
    -   **Low-Light Enhancement Toggle:** Activate or deactivate a low-light enhancement filter (using OpenCV's CLAHE and bilateral filtering) for improved visibility in challenging lighting conditions.
    -   **Name Unidentified:** Assign a name to an unidentified person.
    -   **Edit Name:** Change the name of a recognized person.
-   **Information Panel:** Displays real-time information about recognized individuals, including their name, ID, confidence score, and a reference image.
-   **FPS Counter:** Shows the frames per second (FPS) of the video processing, providing feedback on system performance.

### 5. Camera Handling
-   **Auto-Detection:** The system automatically scans for and detects all available cameras connected to the system (including built-in webcams and external USB cameras).
-   **User Selection:** If multiple cameras are detected, the GUI presents a dialog box, allowing the user to select their preferred camera source, offering flexibility for various hardware setups (e.g., using an Android phone as a webcam).

## Setup and Installation

1.  **Navigate to the project directory:**
    ```bash
    cd /yolo_human_tracker
    ```

2.  **Create a virtual environment (recommended):**
    ```bash
    python3 -m venv venv
    source venv/bin/activate
    ```

3.  **Install the required libraries:**
    ```bash
    pip install -r requirements.txt
    ```

## How to Run

-   **Start the tracker:**
    ```bash
    python gui_tracker.py
    ```
    If multiple cameras are detected, a selection dialog will appear.

## Controls (GUI Buttons)

-   **Stop/Start:** Toggles the video stream.
-   **Pause/Resume:** Pauses or resumes video processing.
-   **FR: On/Off:** Toggles Face Recognition.
-   **Low Light: On/Off:** Toggles Low Light Enhancement.
-   **Name Unidentified:** Opens a dialog to name an unidentified person.
-   **Edit Name:** Opens a dialog to change the name of a recognized person.

## Change Log

-   **v1.6:** Refactored the codebase to centralize the drawing logic. This has resulted in a more streamlined and maintainable codebase, with reduced code duplication.
-   **v1.5:** Fixed a critical bug in the person naming system that caused instability and data corruption. The naming logic is now robust, ensuring that once a person is named, they are permanently and reliably recognized across sessions. Also improved the efficiency of the face recognition process to prevent unnecessary re-identifications.
-   **v1.4:** Reverted to a pure OpenCV application for maximum stability and reliability. Removed NiceGUI and all threading complexities. Controls are now keyboard-based within the OpenCV window.
-   **v1.3:** Refactored to a hybrid GUI: OpenCV window for video display, NiceGUI for controls and logs. Improved stability and performance by ensuring OpenCV GUI calls are on the main thread. Added loading screen to OpenCV window.
-   **v1.2:** Migrated to a NiceGUI web interface with a multi-threaded architecture for smooth video streaming and added a large FPS counter and live logging panel.
-   **v1.1:** Added pause/quit functionality, an FPS counter, and an automatic zoom feature.
-   **v1.0:** Initial implementation with YOLOv8 and basic centroid tracking.
