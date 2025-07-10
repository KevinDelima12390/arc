import sys
import cv2
from PyQt6.QtWidgets import QApplication, QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QSizePolicy, QInputDialog, QGroupBox, QMessageBox
from PyQt6.QtGui import QImage, QPixmap, QIcon
from PyQt6.QtCore import Qt, QTimer, pyqtSignal, QThread
import numpy as np
import time
from human_tracker_backend import HumanTrackerBackend

# Define global constants for the desired video display size.
# These constants make it easy to change the video feed's dimensions in one place,
# ensuring consistency across the GUI.
VIDEO_DISPLAY_WIDTH = 800
VIDEO_DISPLAY_HEIGHT = 600

def detect_available_cameras():
    """
    Scans for and returns a list of available camera indices.
    This function iterates through a common range of camera indices (0 to 9)
    and attempts to open each camera. If a camera can be opened and a frame
    can be successfully read from it, its index is considered valid and added
    to the list of available cameras. This robust check helps in identifying
    functional cameras and avoiding errors with non-existent or problematic indices.
    """
    available_cameras = []
    for i in range(10):  # Check camera indices from 0 to 9
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            # Attempt to read a frame to confirm the camera is truly functional
            ret, frame = cap.read()
            if ret:
                available_cameras.append(i)
            cap.release()  # Release the camera immediately after checking
    return available_cameras

class VideoStreamThread(QThread):
    """
    A QThread subclass responsible for handling the video stream from the camera
    and processing frames using the HumanTrackerBackend. It emits signals to
    update the GUI with new frames, recognized person information, and FPS.
    Running video capture and processing in a separate thread prevents the GUI
    from freezing, ensuring a smooth user experience.
    """
    # Signals emitted by this thread to communicate with the GUI
    change_pixmap_signal = pyqtSignal(QImage)  # Emits a QImage for video display
    update_info_signal = pyqtSignal(dict)      # Emits a dictionary with recognized person info
    update_fps_signal = pyqtSignal(int)        # Emits the current frames per second
    error_signal = pyqtSignal(str)             # Emits error messages
    camera_name_signal = pyqtSignal(str)       # Emits the name of the opened camera

    def __init__(self, backend, camera_index, parent=None):
        """
        Initializes the VideoStreamThread.
        Args:
            backend (HumanTrackerBackend): An instance of the human tracking backend.
            camera_index (int): The index of the camera to open.
            parent (QObject): The parent QObject for this thread (optional).
        """
        super().__init__(parent)
        self._run = True  # Flag to control the thread's main loop
        self.backend = backend
        self.camera_index = camera_index
        self.cap = cv2.VideoCapture(self.camera_index)  # Open the video capture device
        
        if not self.cap.isOpened():
            self._run = False  # Set run flag to False if camera fails to open
            self.error_signal.emit(f"Error: Could not open camera index {self.camera_index}.")
            print(f"Error: Could not open camera index {self.camera_index}.")
        else:
            # Attempt to get the camera's backend name for display in the GUI.
            # This property is not universally supported across all OpenCV backends/platforms.
            try:
                camera_name = self.cap.getBackendName()
            except Exception:
                camera_name = f"Camera {self.camera_index}"  # Fallback name
            self.camera_name_signal.emit(camera_name)  # Emit the camera name to the GUI
            
        self.prev_frame_time = 0  # Used for FPS calculation

    def run(self):
        print("VideoStreamThread: run() started")
        """
        The main loop of the thread. Continuously reads frames from the camera,
        processes them using the backend, and emits signals to update the GUI.
        The loop runs as long as the `_run` flag is True and the backend is not paused.
        """
        while self._run:
            if not self.backend.paused:  # Process frames only if the backend is not paused
                print("VideoStreamThread: Reading frame...")
                ret, frame = self.cap.read()  # Read a frame from the camera
                print("VideoStreamThread: Frame read.")
                if not ret:
                    self.error_signal.emit("Error: Failed to read frame from camera.")
                    print("Error: Failed to read frame from camera.")
                    time.sleep(0.1)  # Small delay before retrying
                    continue 
                
                print("VideoStreamThread: Processing frame...")
                # Process the frame using the HumanTrackerBackend
                display_frame, tracked_objects, known_faces_db = self.backend.process_frame(frame.copy())
                print("VideoStreamThread: Frame processed.")

                # Convert the OpenCV frame (numpy array) to a QImage for display in PyQt6.
                # This involves converting color format and scaling to the predefined display size.
                rgb_image = cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB)
                h, w, ch = rgb_image.shape
                bytes_per_line = ch * w
                convert_to_qt_format = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
                
                # Scale the QImage to the predefined fixed size for consistent display.
                p = convert_to_qt_format.scaled(
                    VIDEO_DISPLAY_WIDTH, 
                    VIDEO_DISPLAY_HEIGHT, 
                    Qt.AspectRatioMode.KeepAspectRatio, 
                    Qt.TransformationMode.SmoothTransformation
                )
                print("VideoStreamThread: Emitting change_pixmap_signal")
                self.change_pixmap_signal.emit(p)  # Emit the scaled QImage

                # Calculate and emit FPS for performance monitoring.
                current_frame_time = time.time()
                fps = 1 / (current_frame_time - self.prev_frame_time) if self.prev_frame_time > 0 else 0
                self.prev_frame_time = current_frame_time
                print("VideoStreamThread: Emitting update_fps_signal")
                self.update_fps_signal.emit(int(fps))

                # Update the info panel with data of the most confidently recognized person.
                recognized_person_data = self.backend.get_recognized_person_data()
                
                if recognized_person_data:
                    face_id = recognized_person_data.get('face_id') 
                    confidence = recognized_person_data.get('face_confidence', 0.0) 
                    person_name = known_faces_db.get(face_id, {}).get('name', 'Unknown')
                    ref_image_bgr = known_faces_db.get(face_id, {}).get('image') 

                    if isinstance(ref_image_bgr, np.ndarray) and ref_image_bgr.size > 0:
                        ref_image_rgb = cv2.cvtColor(ref_image_bgr, cv2.COLOR_BGR2RGB)
                        
                        print("VideoStreamThread: Emitting update_info_signal with image")
                        self.update_info_signal.emit({
                            "name": person_name,
                            "id": face_id,
                            "confidence": f"{confidence:.2f}",
                            "image": ref_image_rgb 
                        })
                    else:
                        print("VideoStreamThread: Emitting update_info_signal without image")
                        self.update_info_signal.emit({
                            "name": person_name,
                            "id": face_id,
                            "confidence": f"{confidence:.2f}",
                            "image": None 
                        })
                else:
                    # If no recognized person, clear the info panel.
                    print("VideoStreamThread: Emitting update_info_signal for N/A")
                    self.update_info_signal.emit({"name": "N/A", "id": "N/A", "confidence": "N/A", "image": None})

            time.sleep(0.03)  # Small delay to prevent excessive CPU usage
        print("VideoStreamThread: run() finished")

    def stop(self):
        """
        Stops the video stream thread. Sets the `_run` flag to False,
        releases the camera, and waits for the thread to terminate.
        """
        self._run = False
        if self.cap.isOpened():
            self.cap.release()  # Release the camera resource
        self.wait()  # Wait for the thread to finish execution

class HumanTrackerGUI(QWidget):
    """
    The main GUI class for the Human Tracking System. It sets up the user
    interface, manages the video stream thread, and handles user interactions
    with the controls.
    """
    def __init__(self):
        """
        Initializes the GUI. This includes setting up the backend, detecting
        available cameras, allowing the user to select a camera if multiple
        are found, and initializing the UI components.
        """
        super().__init__()
        self.backend = HumanTrackerBackend()  # Initialize the human tracking backend
        self.thread = None  # Placeholder for the video stream thread
        
        # --- Camera Detection and Selection ---
        self.available_cameras = detect_available_cameras()  # Find all available cameras
        if not self.available_cameras:
            # If no cameras are found, display an error message and exit the application.
            app = QApplication.instance()
            if not app:
                app = QApplication(sys.argv)
            error_dialog = QMessageBox()
            error_dialog.setIcon(QMessageBox.Icon.Critical)
            error_dialog.setText("No cameras found!")
            error_dialog.setInformativeText("Please ensure a camera is connected and drivers are installed.")
            error_dialog.setWindowTitle("Camera Error")
            error_dialog.exec()
            sys.exit(1)  # Exit the application

        if len(self.available_cameras) > 1:
            # If multiple cameras are found, prompt the user to select one.
            camera_options = [f"Camera {idx}" for idx in self.available_cameras]
            item, ok = QInputDialog.getItem(
                self, "Select Camera", "Choose a camera:",
                camera_options, 0, False
            )
            if ok and item:
                # Set the selected camera index based on user's choice.
                self.camera_index = self.available_cameras[camera_options.index(item)]
            else:
                # If the user cancels the selection, exit the application.
                sys.exit(0)
        else:
            # If only one camera is found, automatically select it.
            self.camera_index = self.available_cameras[0]

        self.init_ui()  # Initialize the UI components
        self.start_video_stream()  # Start the video stream

    def init_ui(self):
        """
        Sets up the main layout and widgets of the GUI. This includes the
        video display area, control buttons, and the recognized person info panel.
        """
        self.setWindowTitle("Human Tracking with Face Recognition")
        # Set the initial window geometry to accommodate video and side panel.
        self.setGeometry(100, 100, VIDEO_DISPLAY_WIDTH + 400, VIDEO_DISPLAY_HEIGHT + 100)

        # Main layout is horizontal, dividing the window into video and right panels.
        main_layout = QHBoxLayout()
        video_panel_layout = QVBoxLayout()  # Layout for video display and camera info
        right_panel_layout = QVBoxLayout()  # Layout for controls and info panel

        # --- Video Display Area ---
        self.video_label = QLabel()  # Label to display the video frames
        self.video_label.setFixedSize(VIDEO_DISPLAY_WIDTH, VIDEO_DISPLAY_HEIGHT)
        self.video_label.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)
        self.video_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.video_label.setStyleSheet("background-color: black; border: 1px solid #333;")
        video_panel_layout.addWidget(self.video_label)

        # --- Camera Info Label ---
        self.camera_info_label = QLabel("Camera: Detecting...")  # Displays the name of the active camera
        self.camera_info_label.setStyleSheet("font-style: italic; color: #aaa;")
        video_panel_layout.addWidget(self.camera_info_label, alignment=Qt.AlignmentFlag.AlignCenter)

        # --- Controls Group ---
        controls_group_box = QGroupBox("Controls")  # Group box for control buttons
        controls_layout = QHBoxLayout()
        controls_group_box.setLayout(controls_layout)

        # Control buttons and their connections to respective methods
        self.start_stop_button = QPushButton("Stop")
        self.start_stop_button.clicked.connect(self.toggle_start_stop)
        controls_layout.addWidget(self.start_stop_button)

        self.pause_button = QPushButton("Pause")
        self.pause_button.clicked.connect(self.toggle_pause)
        controls_layout.addWidget(self.pause_button)

        self.fr_toggle_button = QPushButton("FR: On")
        self.fr_toggle_button.clicked.connect(self.toggle_fr)
        controls_layout.addWidget(self.fr_toggle_button)

        self.ll_toggle_button = QPushButton("Low Light: Off")
        self.ll_toggle_button.clicked.connect(self.toggle_ll)
        controls_layout.addWidget(self.ll_toggle_button)

        self.name_button = QPushButton("Name Unidentified")
        self.name_button.clicked.connect(self.name_person)
        controls_layout.addWidget(self.name_button)

        self.edit_name_button = QPushButton("Edit Name")
        self.edit_name_button.clicked.connect(self.edit_name)
        controls_layout.addWidget(self.edit_name_button)
        
        video_panel_layout.addWidget(controls_group_box)
        main_layout.addLayout(video_panel_layout)

        # --- Right Side Panel (Info and Future Extensions) ---
        
        # Recognized Person Info Group
        info_group_box = QGroupBox("Recognized Person Info")
        info_panel_layout = QVBoxLayout()
        info_group_box.setLayout(info_panel_layout)

        # Labels to display recognized person's details
        self.name_label = QLabel("Name: N/A")
        self.id_label = QLabel("ID: N/A")
        self.confidence_label = QLabel("Confidence: N/A")
        
        self.face_image_label = QLabel()  # Label to display the recognized face image
        self.face_image_label.setFixedSize(120, 120)
        self.face_image_label.setStyleSheet("background-color: #2c2c2c; border: 1px dashed #666; border-radius: 5px;")
        self.face_image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.face_image_label.setText("No Face") 
        
        self.fps_label = QLabel("FPS: 0")  # Label to display FPS
        self.fps_label.setStyleSheet("font-weight: bold; color: #00ff00;") 

        # Add info labels to the info panel layout
        info_panel_layout.addWidget(self.name_label)
        info_panel_layout.addWidget(self.id_label)
        info_panel_layout.addWidget(self.confidence_label)
        info_panel_layout.addWidget(self.face_image_label)
        info_panel_layout.addWidget(self.fps_label)
        info_panel_layout.addStretch()  # Adds a stretchable space at the bottom

        right_panel_layout.addWidget(info_group_box)
        right_panel_layout.addStretch(1)

        self.status_bar = QLabel("Application Ready.")  # Status bar at the bottom of the window
        self.status_bar.setStyleSheet("padding: 5px; background-color: #333; color: white;")
        
        main_layout.addLayout(right_panel_layout)

        # Overall layout combines the main horizontal layout with the status bar.
        self.overall_layout = QVBoxLayout()
        self.overall_layout.addLayout(main_layout)
        self.overall_layout.addWidget(self.status_bar) 

        self.setLayout(self.overall_layout)  # Set the main layout for the window
        

    def start_video_stream(self):
        """
        Starts or restarts the video stream thread. If a thread is already running,
        it is stopped and waited for termination before a new one is created.
        Signals from the new thread are connected to appropriate GUI update methods.
        """
        if self.thread and self.thread.isRunning():
            self.thread.stop()
            self.thread.wait()
        
        # Create and start a new VideoStreamThread with the selected camera index.
        self.thread = VideoStreamThread(self.backend, self.camera_index, parent=self)
        # Connect thread signals to GUI update slots
        self.thread.change_pixmap_signal.connect(self.update_image)
        self.thread.update_info_signal.connect(self.update_info_panel)
        self.thread.update_fps_signal.connect(self.update_fps)
        self.thread.error_signal.connect(self.display_error_message)
        self.thread.camera_name_signal.connect(self.update_camera_name)
        self.thread.start()
        self.status_bar.setText("Video stream started. Initializing models...")

    def update_camera_name(self, name):
        """
        Updates the camera information label in the GUI with the name of the
        currently active camera.
        Args:
            name (str): The name of the camera.
        """
        self.camera_info_label.setText(f"Source: {name}")

    def update_image(self, qt_image):
        """
        Updates the video display label with a new QImage frame.
        Args:
            qt_image (QImage): The QImage to display.
        """
        self.video_label.setPixmap(QPixmap.fromImage(qt_image))

    def update_info_panel(self, info_data):
        """
        Updates the recognized person information panel with data received
        from the video stream thread. This includes name, ID, confidence,
        and a reference image of the face.
        Args:
            info_data (dict): A dictionary containing recognized person's data.
        """
        self.name_label.setText(f"Name: {info_data['name']}")
        self.id_label.setText(f"ID: {info_data['id']}")
        self.confidence_label.setText(f"Confidence: {info_data['confidence']}")
        
        if info_data['image'] is not None and isinstance(info_data['image'], np.ndarray):
            try:
                # Convert numpy array image to QPixmap for display.
                h, w, ch = info_data['image'].shape
                bytes_per_line = ch * w
                qt_image = QImage(info_data['image'].data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
                self.face_image_label.setPixmap(QPixmap.fromImage(qt_image).scaled(
                    self.face_image_label.width(), self.face_image_label.height(), 
                    Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation
                ))
            except Exception as e:
                print(f"Error converting image for info panel: {e}")
                self.face_image_label.clear()
                self.face_image_label.setText("Image Error")
                self.display_error_message(f"Image conversion error: {e}") 
        else:
            self.face_image_label.clear()
            self.face_image_label.setText("No Face")

    def toggle_start_stop(self):
        """
        Toggles the start/stop state of the video stream. When stopped,
        it clears the video display and info panel.
        """
        if self.thread and self.thread._run:
            self.thread.stop()
            self.start_stop_button.setText("Start")
            # Disable control buttons when stream is stopped
            self.pause_button.setEnabled(False)
            self.fr_toggle_button.setEnabled(False)
            self.ll_toggle_button.setEnabled(False)
            self.name_button.setEnabled(False)
            self.edit_name_button.setEnabled(False) # Disable edit name button
            self.video_label.clear()
            self.update_info_panel({"name": "N/A", "id": "N/A", "confidence": "N/A", "image": None}) 
            self.fps_label.setText("FPS: 0")
            self.status_bar.setText("Video stream stopped.")
        else:
            self.start_video_stream() 
            self.start_stop_button.setText("Stop")
            # Enable control buttons when stream is started
            self.pause_button.setEnabled(True)
            self.fr_toggle_button.setEnabled(True)
            self.ll_toggle_button.setEnabled(True)
            self.name_button.setEnabled(True)
            self.edit_name_button.setEnabled(True) # Enable edit name button
            self.status_bar.setText("Video stream started.")

    def toggle_pause(self):
        """
        Toggles the pause/resume state of the video processing in the backend.
        Updates the button text and status bar accordingly.
        """
        self.backend.paused = not self.backend.paused
        if self.backend.paused:
            self.pause_button.setText("Resume")
            self.status_bar.setText("Video stream paused.")
        else:
            self.pause_button.setText("Pause")
            self.status_bar.setText("Video stream resumed.")

    def toggle_fr(self):
        """
        Toggles the face recognition feature in the backend.
        Updates the button text and status bar.
        """
        status = self.backend.toggle_face_recognition()
        self.fr_toggle_button.setText(f"FR: {'On' if status else 'Off'}")
        self.status_bar.setText(f"Face Recognition: {'Enabled' if status else 'Disabled'}.")

    def toggle_ll(self):
        """
        Toggles the low-light enhancement feature in the backend.
        Updates the button text and status bar.
        """
        status = self.backend.toggle_low_light_enhancement()
        self.ll_toggle_button.setText(f"Low Light: {'On' if status else 'Off'}")
        self.status_bar.setText(f"Low Light Enhancement: {'Enabled' if status else 'Disabled'}.")

    def update_fps(self, fps):
        """
        Updates the FPS display label.
        Args:
            fps (int): The current frames per second.
        """
        self.fps_label.setText(f"FPS: {fps}")

    def name_person(self):
        """
        Handles the "Name Unidentified" button click. Prompts the user to select
        an unidentified person and enter a new name. Calls the backend to
        rename the person.
        """
        unidentified_face_ids = self.backend.get_unidentified_face_ids()
        if unidentified_face_ids:
            item, ok = QInputDialog.getItem(
                self, "Name Unidentified Person", "Select an Unidentified Person ID:", 
                unidentified_face_ids, 0, False
            )
            if ok and item:
                target_face_id = item
                new_name, ok_name = QInputDialog.getText(self, "Enter Name", f"Enter a name for {target_face_id}:")
                if ok_name and new_name and not new_name.isspace():
                    success, message = self.backend.name_unidentified_person(target_face_id, new_name)
                    if success:
                        self.status_bar.setText(message)
                    else:
                        self.display_error_message(message)
        else:
            self.status_bar.setText("No 'Unidentified' person currently tracked to name.")

    def edit_name(self):
        """
        Handles the "Edit Name" button click. Prompts the user to select a
        known person and enter a new name. Calls the backend to edit the person's name.
        """
        known_names = self.backend.get_known_person_names()
        if known_names:
            old_name, ok = QInputDialog.getItem(
                self, "Edit Person's Name", "Select a person to rename:",
                known_names, 0, False
            )
            if ok and old_name:
                new_name, ok_name = QInputDialog.getText(self, "Enter New Name", f"Enter a new name for {old_name}:")
                if ok_name and new_name and not new_name.isspace():
                    success, message = self.backend.edit_person_name(old_name, new_name)
                    if success:
                        self.status_bar.setText(message)
                    else:
                        self.display_error_message(message)
        else:
            self.status_bar.setText("No named people in the database to edit.")
            
    def display_error_message(self, message):
        """
        Displays an error message in the status bar, temporarily changing its
        background color to red for emphasis.
        Args:
            message (str): The error message to display.
        """
        self.status_bar.setText(f"ERROR: {message}")
        self.status_bar.setStyleSheet("padding: 5px; background-color: #8B0000; color: white;") 
        # Revert status bar style after 5 seconds
        QTimer.singleShot(5000, lambda: self.status_bar.setStyleSheet("padding: 5px; background-color: #333; color: white;"))

    def closeEvent(self, event):
        """
        Handles the window close event. Ensures the backend saves known faces
        and the video stream thread is properly stopped before the application exits.
        """
        print("Application closing. Stopping video stream and saving backend data...")
        self.backend.save_known_faces()  # Save known faces before exiting
        if self.thread and self.thread.isRunning(): 
            self.thread.stop() 
            self.thread.wait() 
        event.accept()  # Accept the close event

if __name__ == "__main__":
    # Main entry point of the application.
    # Creates a QApplication instance, initializes the GUI, shows it, and starts the event loop.
    app = QApplication(sys.argv)
    gui = HumanTrackerGUI()
    gui.show()
    sys.exit(app.exec())
