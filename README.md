# AI Human Tracking System

This project is a real-time human detection and tracking system using a webcam and a pure OpenCV GUI.

## Status

**Version 1.4 is stable and working.** This version prioritizes stability and direct control over web-based features.

## Core Features

-   **Pure OpenCV GUI:** All video display and controls are handled directly within an OpenCV window.
-   **Real-time Tracking:** Uses YOLOv8 and a centroid tracker to identify and follow people.
-   **Hardware Accelerated:** Leverages Apple MPS for significant performance gains on M1/M2/M3 Macs.
-   **FPS Counter:** Displays the real-time processing speed directly on the video feed.
-   **Basic Controls:** Pause/Resume and Quit functionality via keyboard input.

## Setup and Installation

1.  **Navigate to the project directory:**
    ```bash
    cd /Users/kevinarthurdelima/Desktop/test2/yolo_human_tracker
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
    python tracker.py
    ```

## Controls

-   **`q`**: Quit the application.
-   **`p`**: Pause/Resume the video feed.

## Change Log

-   **v1.0:** Initial implementation with YOLOv8 and basic centroid tracking.
-   **v1.1:** Added pause/quit functionality, an FPS counter, and an automatic zoom feature.
-   **v1.2:** Migrated to a NiceGUI web interface with a multi-threaded architecture for smooth video streaming and added a large FPS counter and live logging panel.
-   **v1.3:** Refactored to a hybrid GUI: OpenCV window for video display, NiceGUI for controls and logs. Improved stability and performance by ensuring OpenCV GUI calls are on the main thread. Added loading screen to OpenCV window.
## Change Log

-   **v1.5:** Fixed a critical bug in the person naming system that caused instability and data corruption. The naming logic is now robust, ensuring that once a person is named, they are permanently and reliably recognized across sessions. Also improved the efficiency of the face recognition process to prevent unnecessary re-identifications.
-   **v1.4:** Reverted to a pure OpenCV application for maximum stability and reliability. Removed NiceGUI and all threading complexities. Controls are now keyboard-based within the OpenCV window.