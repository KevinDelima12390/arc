import sys
import cv2
from PyQt6.QtWidgets import QApplication, QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QSizePolicy, QInputDialog, QGroupBox
from PyQt6.QtGui import QImage, QPixmap, QIcon
from PyQt6.QtCore import Qt, QTimer, pyqtSignal, QThread
import numpy as np
import time
from human_tracker_backend import HumanTrackerBackend

# Define a global constant for the desired video display size
# This makes it easy to change in one place
VIDEO_DISPLAY_WIDTH = 800
VIDEO_DISPLAY_HEIGHT = 600

class VideoStreamThread(QThread):
    change_pixmap_signal = pyqtSignal(QImage)
    update_info_signal = pyqtSignal(dict)
    update_fps_signal = pyqtSignal(int)
    error_signal = pyqtSignal(str) 

    def __init__(self, backend, parent=None):
        super().__init__(parent) 
        self._run = True
        self.backend = backend
        self.cap = cv2.VideoCapture(0) 
        if not self.cap.isOpened():
            self._run = False
            self.error_signal.emit("Error: Could not open video stream. Check camera connection.")
            print("Error: Could not open video stream.")
        self.prev_frame_time = 0

    def run(self):
        while self._run:
            if not self.backend.paused:
                ret, frame = self.cap.read()
                if not ret:
                    self.error_signal.emit("Error: Failed to read frame from camera.")
                    print("Error: Failed to read frame from camera.")
                    time.sleep(0.1) 
                    continue 
                
                processed_frame, tracked_objects, known_faces_db = self.backend.process_frame(frame.copy())

                # Draw bounding boxes and labels on the processed frame
                for person_id, obj_data in tracked_objects.items():
                    x1, y1, x2, y2 = obj_data['box']
                    face_id_label = obj_data.get('face_id', 'Unknown')
                    face_confidence = obj_data.get('face_confidence', 0.0)
                    
                    person_name = known_faces_db.get(face_id_label, {}).get('name', str(face_id_label))
                    
                    label = f"P:{person_id} ({person_name})"
                    
                    if isinstance(face_id_label, str) and not face_id_label.startswith("Unidentified_") and face_id_label != "No Face Detected" and person_name != str(face_id_label):
                        label += f" C:{face_confidence:.2f}"
                        color = (0, 255, 0) # Green for recognized
                    elif face_id_label.startswith("Unidentified_"):
                        label = f"P:{person_id} (Unidentified)"
                        color = (0, 165, 255) # Orange for unidentified
                    else:
                        color = (0, 0, 255) # Red for unknown/no face (Person ID only)

                    cv2.rectangle(processed_frame, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(processed_frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

                    face_rect = obj_data.get('face_rect')
                    if face_rect:
                        top, right, bottom, left = face_rect
                        fx1_abs, fy1_abs = x1 + left, y1 + top
                        fx2_abs, fy2_abs = x1 + right, y1 + bottom
                        cv2.rectangle(processed_frame, (fx1_abs, fy1_abs), (fx2_abs, fy2_abs), (255, 0, 0), 2) # Blue for face

                # Convert to QImage for main video display
                rgb_image = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)
                h, w, ch = rgb_image.shape
                bytes_per_line = ch * w
                convert_to_qt_format = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
                
                # --- CHANGE HERE: Scale to the predefined fixed size ---
                p = convert_to_qt_format.scaled(
                    VIDEO_DISPLAY_WIDTH, 
                    VIDEO_DISPLAY_HEIGHT, 
                    Qt.AspectRatioMode.KeepAspectRatio, 
                    Qt.TransformationMode.SmoothTransformation
                )
                self.change_pixmap_signal.emit(p)

                # Calculate and emit FPS
                current_frame_time = time.time()
                fps = 1 / (current_frame_time - self.prev_frame_time) if self.prev_frame_time > 0 else 0
                self.prev_frame_time = current_frame_time
                self.update_fps_signal.emit(int(fps))

                # --- Update info panel data ---
                recognized_person_data = self.backend.get_recognized_person_data()
                
                if recognized_person_data:
                    face_id = recognized_person_data.get('face_id') 
                    confidence = recognized_person_data.get('face_confidence', 0.0) 
                    person_name = known_faces_db.get(face_id, {}).get('name', 'Unknown')
                    ref_image_bgr = known_faces_db.get(face_id, {}).get('image') 

                    if isinstance(ref_image_bgr, np.ndarray) and ref_image_bgr.size > 0:
                        ref_image_rgb = cv2.cvtColor(ref_image_bgr, cv2.COLOR_BGR2RGB)
                        
                        self.update_info_signal.emit({
                            "name": person_name,
                            "id": face_id,
                            "confidence": f"{confidence:.2f}",
                            "image": ref_image_rgb 
                        })
                    else:
                        self.update_info_signal.emit({
                            "name": person_name,
                            "id": face_id,
                            "confidence": f"{confidence:.2f}",
                            "image": None 
                        })
                else:
                    self.update_info_signal.emit({"name": "N/A", "id": "N/A", "confidence": "N/A", "image": None})

            time.sleep(0.03)

    def stop(self):
        self._run = False
        if self.cap.isOpened():
            self.cap.release()
        self.wait()

class HumanTrackerGUI(QWidget):
    def __init__(self):
        super().__init__()
        self.backend = HumanTrackerBackend()
        self.thread = None 
        self.init_ui()
        self.start_video_stream() 

    def init_ui(self):
        self.setWindowTitle("Human Tracking with Face Recognition")
        # Adjust overall window size to accommodate fixed video size + side panel
        self.setGeometry(100, 100, VIDEO_DISPLAY_WIDTH + 400, VIDEO_DISPLAY_HEIGHT + 100) 

        main_layout = QHBoxLayout()
        video_panel_layout = QVBoxLayout()
        right_panel_layout = QVBoxLayout()

        # --- Video Display Area ---
        self.video_label = QLabel()
        # --- CHANGE HERE: Set fixed size and policy ---
        self.video_label.setFixedSize(VIDEO_DISPLAY_WIDTH, VIDEO_DISPLAY_HEIGHT) 
        self.video_label.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed) 
        self.video_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.video_label.setStyleSheet("background-color: black; border: 1px solid #333;") 
        video_panel_layout.addWidget(self.video_label)

        # --- Controls Group ---
        controls_group_box = QGroupBox("Controls")
        controls_layout = QHBoxLayout()
        controls_group_box.setLayout(controls_layout)

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

        self.name_button = QPushButton("Name Person")
        self.name_button.clicked.connect(self.name_person)
        controls_layout.addWidget(self.name_button)
        
        video_panel_layout.addWidget(controls_group_box)
        main_layout.addLayout(video_panel_layout) # No stretch factor needed if video label is fixed

        # --- Right Side Panel (Info and Future Extensions) ---
        
        # Recognized Person Info Group
        info_group_box = QGroupBox("Recognized Person Info")
        info_panel_layout = QVBoxLayout()
        info_group_box.setLayout(info_panel_layout)

        self.name_label = QLabel("Name: N/A")
        self.id_label = QLabel("ID: N/A")
        self.confidence_label = QLabel("Confidence: N/A")
        
        self.face_image_label = QLabel()
        self.face_image_label.setFixedSize(120, 120) 
        self.face_image_label.setStyleSheet("background-color: #2c2c2c; border: 1px dashed #666; border-radius: 5px;")
        self.face_image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.face_image_label.setText("No Face") 
        
        self.fps_label = QLabel("FPS: 0")
        self.fps_label.setStyleSheet("font-weight: bold; color: #00ff00;") 

        info_panel_layout.addWidget(self.name_label)
        info_panel_layout.addWidget(self.id_label)
        info_panel_layout.addWidget(self.confidence_label)
        info_panel_layout.addWidget(self.face_image_label)
        info_panel_layout.addWidget(self.fps_label)
        info_panel_layout.addStretch() 
        
        right_panel_layout.addWidget(info_group_box)
        right_panel_layout.addStretch(1) 

        self.status_bar = QLabel("Application Ready.") 
        self.status_bar.setStyleSheet("padding: 5px; background-color: #333; color: white;")
        
        main_layout.addLayout(right_panel_layout) # No stretch factor needed here either

        self.overall_layout = QVBoxLayout()
        self.overall_layout.addLayout(main_layout)
        self.overall_layout.addWidget(self.status_bar) 

        self.setLayout(self.overall_layout)
        

    def start_video_stream(self):
        if self.thread and self.thread.isRunning():
            self.thread.stop() 
            self.thread.wait()
        
        self.thread = VideoStreamThread(self.backend, parent=self) 
        self.thread.change_pixmap_signal.connect(self.update_image)
        self.thread.update_info_signal.connect(self.update_info_panel)
        self.thread.update_fps_signal.connect(self.update_fps)
        self.thread.error_signal.connect(self.display_error_message) 
        self.thread.start()
        self.status_bar.setText("Video stream started. Initializing models...")


    def update_image(self, qt_image):
        self.video_label.setPixmap(QPixmap.fromImage(qt_image))

    def update_info_panel(self, info_data):
        self.name_label.setText(f"Name: {info_data['name']}")
        self.id_label.setText(f"ID: {info_data['id']}")
        self.confidence_label.setText(f"Confidence: {info_data['confidence']}")
        
        if info_data['image'] is not None and isinstance(info_data['image'], np.ndarray):
            try:
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
        if self.thread and self.thread._run:
            self.thread.stop()
            self.start_stop_button.setText("Start")
            self.pause_button.setEnabled(False)
            self.fr_toggle_button.setEnabled(False)
            self.ll_toggle_button.setEnabled(False)
            self.name_button.setEnabled(False)
            self.video_label.clear()
            self.update_info_panel({"name": "N/A", "id": "N/A", "confidence": "N/A", "image": None}) 
            self.fps_label.setText("FPS: 0")
            self.status_bar.setText("Video stream stopped.")
        else:
            self.start_video_stream() 
            self.start_stop_button.setText("Stop")
            self.pause_button.setEnabled(True)
            self.fr_toggle_button.setEnabled(True)
            self.ll_toggle_button.setEnabled(True)
            self.name_button.setEnabled(True)
            self.status_bar.setText("Video stream started.")

    def toggle_pause(self):
        self.backend.paused = not self.backend.paused
        if self.backend.paused:
            self.pause_button.setText("Resume")
            self.status_bar.setText("Video stream paused.")
        else:
            self.pause_button.setText("Pause")
            self.status_bar.setText("Video stream resumed.")

    def toggle_fr(self):
        status = self.backend.toggle_face_recognition()
        self.fr_toggle_button.setText(f"FR: {'On' if status else 'Off'}")
        self.status_bar.setText(f"Face Recognition: {'Enabled' if status else 'Disabled'}.")

    def toggle_ll(self):
        status = self.backend.toggle_low_light_enhancement()
        self.ll_toggle_button.setText(f"Low Light: {'On' if status else 'Off'}")
        self.status_bar.setText(f"Low Light Enhancement: {'Enabled' if status else 'Disabled'}.")

    def update_fps(self, fps):
        self.fps_label.setText(f"FPS: {fps}")

    def name_person(self):
        unidentified_face_ids = self.backend.get_unidentified_face_ids()
        if unidentified_face_ids:
            item, ok = QInputDialog.getItem(
                self, "Name Person", "Select an Unidentified Person ID:", 
                unidentified_face_ids, 0, False
            )
            if ok and item:
                target_face_id = item
                new_name, ok_name = QInputDialog.getText(self, "Name Person", f"Enter a name for {target_face_id}:")
                if ok_name and new_name:
                    if self.backend.name_unidentified_person(target_face_id, new_name):
                        print(f"Assigned name '{new_name}' to face ID {target_face_id}.")
                        self.status_bar.setText(f"Assigned '{new_name}' to ID '{target_face_id}'.")
                    else:
                        print(f"Error: Face ID {target_face_id} not found in database or could not be named.")
                        self.display_error_message(f"Error: Could not name ID '{target_face_id}'. Check backend logs.")
        else:
            print("No 'Unidentified' person currently tracked to name.")
            self.status_bar.setText("No 'Unidentified' person to name.")
            
    def display_error_message(self, message):
        self.status_bar.setText(f"ERROR: {message}")
        self.status_bar.setStyleSheet("padding: 5px; background-color: #8B0000; color: white;") 
        QTimer.singleShot(5000, lambda: self.status_bar.setStyleSheet("padding: 5px; background-color: #333; color: white;"))


    def closeEvent(self, event):
        print("Application closing. Stopping video stream and saving backend data...")
        self.backend.save_known_faces() 
        if self.thread and self.thread.isRunning(): 
            self.thread.stop() 
            self.thread.wait() 
        event.accept()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    gui = HumanTrackerGUI()
    gui.show()
    sys.exit(app.exec())