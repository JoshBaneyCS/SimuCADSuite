import sys
import os
import random
import csv
from PyQt5.QtWidgets import (QApplication, QMainWindow, QVBoxLayout, QLabel, QPushButton, QWidget,
                             QFileDialog, QAction, QMessageBox, QComboBox, QLineEdit, QCheckBox, QDialog)
from PyQt5.QtGui import QIcon, QPixmap, QFont
from PyQt5.QtCore import QTimer, Qt

SETTINGS_FILE = "settings.txt"

UNIT_MAPPING = {
    "m/s": 1,
    "ft/s": 2,
    "mph": 3,
    "km/h": 4,
    "Knots": 5
}

class LoadingScreen(QDialog):
    def __init__(self, parent=None, logo_path=None):
        super().__init__(parent)
        self.setWindowTitle("Loading SimuCADSuite")
        self.setFixedSize(600, 600)
        self.setWindowFlags(Qt.Dialog | Qt.CustomizeWindowHint)

        layout = QVBoxLayout()

        if logo_path:
            pixmap = QPixmap(logo_path).scaled(500, 500, Qt.KeepAspectRatio)
            logo_label = QLabel()
            logo_label.setPixmap(pixmap)
            layout.addWidget(logo_label, alignment=Qt.AlignCenter)

        title_label = QLabel("SimuCADSuite v0.1.0")
        title_label.setAlignment(Qt.AlignCenter)
        title_label.setStyleSheet("font-size: 20px;")
        layout.addWidget(title_label)

        self.message_label = QLabel("Loading packages...")
        self.message_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.message_label)

        self.setLayout(layout)

        self.messages = [
            "Loading packages...",
            "Preparing particle accelerator...",
            "Downloading more RAM...",
            "Initializing MATPLOTLIB...",
            "Initializing SciPy...",
            "Warming up the flux capacitor...",
            "Optimizing machine learning models...",
            "Compiling shaders...",
            "Calibrating sensors...",
            "Tuning neural networks..."
        ]

        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_message)
        self.timer.start(random.randint(500, 1500))

        # Center the loading screen on the user's screen
        self.center()

    def update_message(self):
        next_message = random.choice(self.messages)
        self.message_label.setText(next_message)

    def center(self):
        screen_geometry = QApplication.desktop().screenGeometry()
        screen_width = screen_geometry.width()
        screen_height = screen_geometry.height()
        x = int((screen_width - self.width()) / 2)
        y = int((screen_height - self.height()) / 2)
        self.move(x, y)

class MainWindow(QMainWindow):
    def __init__(self, logo_path=None, favicon_path=None):
        super().__init__()

        self.setWindowTitle("SimuCADSuite v0.1.0 - By Josh Baney")
        self.setWindowIcon(QIcon(favicon_path))
        self.setGeometry(100, 100, 800, 600)
        self.center_window()

        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)

        self.layout = QVBoxLayout()
        self.central_widget.setLayout(self.layout)

        self.kinematics_path = ""
        self.fluid_dynamics_path = ""
        self.audio_waveform_path = ""
        self.CAD_path = ""
        self.velocity = ""
        self.dark_mode = False
        self.multithreading = False
        self.gpu_rendering = False

        self.previous_page = None

        # Initialize pages
        self.create_pages()

        self.create_menu()

        self.load_settings()  # Load settings, including dark mode
        self.apply_dark_mode_if_enabled()  # Apply dark mode immediately after loading settings

        # Apply custom stylesheet to make elements larger
        self.apply_stylesheet()

    def create_menu(self):
        menubar = self.menuBar()

        file_menu = menubar.addMenu("File")
        save_action = QAction("Save", self)
        save_as_action = QAction("Save As...", self)
        settings_action = QAction("Settings", self)
        load_action = QAction("Load", self)
        about_action = QAction("About", self)

        file_menu.addAction(save_action)
        file_menu.addAction(save_as_action)
        file_menu.addAction(settings_action)
        file_menu.addAction(load_action)
        file_menu.addAction(about_action)

        about_action.triggered.connect(self.show_about)
        settings_action.triggered.connect(self.show_settings)

        # Add Back and Home buttons
        back_action = QAction("Back", self)
        home_action = QAction("Home", self)

        menubar.addAction(back_action)
        menubar.addAction(home_action)

        back_action.triggered.connect(self.go_back)
        home_action.triggered.connect(self.show_home_page)

    def create_pages(self):
        # Create Home Page
        self.home_widget = QWidget(self)
        self.home_layout = QVBoxLayout(self.home_widget)

        self.kinematics_button = QPushButton("Kinematics Simulator")
        self.fluid_dynamics_button = QPushButton("Fluid Dynamics Simulator")
        self.audio_waveform_button = QPushButton("Audio Waveform Decoder")

        self.kinematics_button.clicked.connect(self.show_kinematics)
        self.fluid_dynamics_button.clicked.connect(self.show_fluid_dynamics)
        self.audio_waveform_button.clicked.connect(self.show_audio_decoder)

        self.home_layout.addWidget(self.kinematics_button)
        self.home_layout.addWidget(self.fluid_dynamics_button)
        self.home_layout.addWidget(self.audio_waveform_button)

        # Create Kinematics Page
        self.kinematics_widget = QWidget(self)
        self.kinematics_layout = QVBoxLayout(self.kinematics_widget)
        self.create_kinematics_page(self.kinematics_layout)

        # Create Fluid Dynamics Page
        self.fluid_dynamics_widget = QWidget(self)
        self.fluid_dynamics_layout = QVBoxLayout(self.fluid_dynamics_widget)
        self.create_fluid_dynamics_page(self.fluid_dynamics_layout)

        # Add all widgets to the layout but only show the home page initially
        self.layout.addWidget(self.home_widget)
        self.home_widget.setVisible(True)
        self.layout.addWidget(self.kinematics_widget)
        self.kinematics_widget.setVisible(False)
        self.layout.addWidget(self.fluid_dynamics_widget)
        self.fluid_dynamics_widget.setVisible(False)

    def create_kinematics_page(self, layout):
        self.launch_angle = QLineEdit()
        self.initial_velocity = QLineEdit()
        self.gravity = QLineEdit()
        self.surface_area = QLineEdit()
        self.initial_height = QLineEdit()

        self.angle_unit = QComboBox()
        self.angle_unit.addItems(["Degrees", "Radians"])

        self.velocity_unit = QComboBox()
        self.velocity_unit.addItems(["m/s", "ft/s", "mph", "km/h"])

        self.gravity_unit = QComboBox()
        self.gravity_unit.addItems(["m/s²", "ft/s²", "g"])

        self.area_unit = QComboBox()
        self.area_unit.addItems(["m²", "ft²", "cm²", "in²"])

        self.height_unit = QComboBox()
        self.height_unit.addItems(["m", "ft", "cm", "in"])

        layout.addWidget(QLabel("Launch Angle:"))
        layout.addWidget(self.launch_angle)
        layout.addWidget(self.angle_unit)

        layout.addWidget(QLabel("Initial Velocity:"))
        layout.addWidget(self.initial_velocity)
        layout.addWidget(self.velocity_unit)

        layout.addWidget(QLabel("Acceleration due to Gravity:"))
        layout.addWidget(self.gravity)
        layout.addWidget(self.gravity_unit)

        layout.addWidget(QLabel("Surface Area of the Object:"))
        layout.addWidget(self.surface_area)
        layout.addWidget(self.area_unit)

        layout.addWidget(QLabel("Initial Height:"))
        layout.addWidget(self.initial_height)
        layout.addWidget(self.height_unit)

        save_button = QPushButton("Save and Go")
        save_button.clicked.connect(self.save_kinematics_data)
        layout.addWidget(save_button)

    def create_fluid_dynamics_page(self, layout):
        # CAD file selection
        self.cad_path_label = QLabel("No file selected.")
        select_cad_button = QPushButton("Select CAD File")
        select_cad_button.clicked.connect(self.select_cad_file)

        # Increase button and text input sizes
        select_cad_button.setStyleSheet("font-size: 18px; padding: 10px;")
        self.cad_path_label.setStyleSheet("font-size: 16px;")

        # Set QLabel fonts
        font = QFont()
        font.setPointSize(18)

        cad_file_label = QLabel("CAD File:")
        cad_file_label.setFont(font)
        velocity_label = QLabel("Relative Velocity:")
        velocity_label.setFont(font)

        layout.addWidget(cad_file_label)
        layout.addWidget(self.cad_path_label)
        layout.addWidget(select_cad_button)

        # Velocity input
        self.velocity_input = QLineEdit()
        self.velocity_input.setStyleSheet("font-size: 18px; padding: 10px;")
        self.velocity_unit = QComboBox()
        self.velocity_unit.addItems(["m/s", "ft/s", "mph", "km/h", "Knots"])
        self.velocity_unit.setStyleSheet("font-size: 18px; padding: 10px;")

        layout.addWidget(velocity_label)
        layout.addWidget(self.velocity_input)
        layout.addWidget(self.velocity_unit)

        # Save and execute button
        save_button = QPushButton("Save and Execute")
        save_button.setStyleSheet("font-size: 18px; padding: 10px;")
        save_button.clicked.connect(self.save_and_execute_fluid_dynamics)
        layout.addWidget(save_button)

    def apply_stylesheet(self):
        # Custom stylesheet to increase the size of buttons, text entries, and dropdowns
        self.setStyleSheet("""
            QPushButton {
                font-size: 16px;
                padding: 10px;
            }
            QComboBox {
                font-size: 16px;
                padding: 10px;
            }
            QLineEdit {
                font-size: 16px;
                padding: 10px;
            }
            QLabel {
                font-size: 16px;
            }
        """)

    def show_home_page(self):
        self.previous_page = "Home"
        self.kinematics_widget.setVisible(False)
        self.fluid_dynamics_widget.setVisible(False)
        self.home_widget.setVisible(True)

    def show_kinematics(self):
        self.previous_page = "Kinematics"
        self.home_widget.setVisible(False)
        self.fluid_dynamics_widget.setVisible(False)
        self.kinematics_widget.setVisible(True)

    def show_fluid_dynamics(self):
        self.previous_page = "FluidDynamics"
        self.home_widget.setVisible(False)
        self.kinematics_widget.setVisible(False)
        self.fluid_dynamics_widget.setVisible(True)

    def show_audio_decoder(self):
        self.previous_page = "AudioDecoder"
        QMessageBox.information(self, "Audio Waveform", "Audio Waveform Decoder coming soon.")

    def go_back(self):
        if self.previous_page == "Home":
            self.show_home_page()
        elif self.previous_page == "Kinematics":
            self.show_kinematics()
        elif self.previous_page == "FluidDynamics":
            self.show_fluid_dynamics()

    def show_about(self):
        about_text = (
            "SimuCADSuite v1.0<br><br>"
            "Written by Joshua Baney<br>"
            "Computer Science undergraduate student<br>"
            "MIT Open Software License<br><br>"
            "Useful Links:<br>"
            "<a href='https://www.joshbaney.com'>Website</a><br>"
            "<a href='https://github.com/JoshBaneyCS'>GitHub</a><br>"
            "<a href='https://www.linkedin.com/in/joshua-baney-34768384/'>LinkedIn</a><br>"
        )

        about_dialog = QDialog(self)
        about_dialog.setWindowTitle("About SimuCADSuite")
        about_layout = QVBoxLayout()

        about_label = QLabel()
        about_label.setText(about_text)
        about_label.setOpenExternalLinks(True)  # Make the links clickable
        about_label.setTextInteractionFlags(Qt.TextBrowserInteraction)  # Allow clicking the links
        about_layout.addWidget(about_label)

        close_button = QPushButton("Close")
        close_button.clicked.connect(about_dialog.accept)
        about_layout.addWidget(close_button)

        about_dialog.setLayout(about_layout)
        about_dialog.exec_()

    def show_settings(self):
        settings_dialog = QDialog(self)
        settings_dialog.setWindowTitle("Settings")
        settings_dialog.setGeometry(100, 100, 400, 600)

        layout = QVBoxLayout()

        # Light/Dark Mode Toggle
        dark_mode_check = QCheckBox("Dark Mode")
        dark_mode_check.setChecked(self.dark_mode)
        dark_mode_check.toggled.connect(self.toggle_dark_mode)
        layout.addWidget(dark_mode_check)

        # Multithreading Toggle
        multithreading_check = QCheckBox("Enable Multi-threading")
        multithreading_check.setChecked(self.multithreading)
        layout.addWidget(multithreading_check)

        # GPU Rendering Button
        gpu_render_check = QCheckBox("Enable GPU Rendering")
        gpu_render_check.setChecked(self.gpu_rendering)
        layout.addWidget(gpu_render_check)

        # Kinematics File Path
        layout.addWidget(QLabel("Kinematics Data Path:"))
        kinematics_path_button = QPushButton("Select Kinematics Data Path")
        kinematics_path_button.clicked.connect(lambda: self.select_directory("kinematics"))
        layout.addWidget(kinematics_path_button)
        self.kinematics_path_label = QLabel(self.kinematics_path)
        layout.addWidget(self.kinematics_path_label)

        # Fluid Dynamics File Path
        layout.addWidget(QLabel("Fluid Dynamics Data Path:"))
        fluid_dynamics_path_button = QPushButton("Select Fluid Dynamics Data Path")
        fluid_dynamics_path_button.clicked.connect(lambda: self.select_directory("fluid_dynamics"))
        layout.addWidget(fluid_dynamics_path_button)
        self.fluid_dynamics_path_label = QLabel(self.fluid_dynamics_path)
        layout.addWidget(self.fluid_dynamics_path_label)

        # Audio Waveform File Path
        layout.addWidget(QLabel("Audio Waveform Data Path:"))
        audio_waveform_path_button = QPushButton("Select Audio Waveform Data Path")
        audio_waveform_path_button.clicked.connect(lambda: self.select_directory("audio_waveform"))
        layout.addWidget(audio_waveform_path_button)
        self.audio_waveform_path_label = QLabel(self.audio_waveform_path)
        layout.addWidget(self.audio_waveform_path_label)

        # Save Settings Button
        save_button = QPushButton("Save Settings")
        save_button.clicked.connect(lambda: self.save_and_close_settings(settings_dialog))
        layout.addWidget(save_button)

        settings_dialog.setLayout(layout)
        settings_dialog.exec_()

    def select_directory(self, module):
        directory = QFileDialog.getExistingDirectory(self, "Select Directory")
        if directory:
            if module == "kinematics":
                self.kinematics_path = directory
                self.kinematics_path_label.setText(directory)
            elif module == "fluid_dynamics":
                self.fluid_dynamics_path = directory
                self.fluid_dynamics_path_label.setText(directory)
            elif module == "audio_waveform":
                self.audio_waveform_path = directory
                self.audio_waveform_path_label.setText(directory)

    def select_cad_file(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "Select CAD File", "", "CAD Files (*.stp *.step *.igs *.iges)")
        if file_path:
            self.CAD_path = file_path
            self.cad_path_label.setText(os.path.basename(file_path))

    def toggle_dark_mode(self, enabled):
        self.dark_mode = enabled
        self.apply_dark_mode_if_enabled()

    def apply_dark_mode_if_enabled(self):
        if self.dark_mode:
            self.setStyleSheet("background-color: #2e2e2e; color: #ffffff;")
        else:
            self.setStyleSheet("")

    def save_and_close_settings(self, dialog):
        self.save_settings()
        dialog.accept()

    def save_settings(self):
        settings = {
            "kinematics_path": self.kinematics_path,
            "fluid_dynamics_path": self.fluid_dynamics_path,
            "audio_waveform_path": self.audio_waveform_path,
            "CAD_path": self.CAD_path,
            "velocity": self.velocity,
            "dark_mode": self.dark_mode,
            "multithreading": self.multithreading,
            "gpu_rendering": self.gpu_rendering
        }
        with open(SETTINGS_FILE, "w") as f:
            for key, value in settings.items():
                f.write(f"{key}={value}\n")

    def load_settings(self):
        if os.path.exists(SETTINGS_FILE):
            with open(SETTINGS_FILE, "r") as f:
                for line in f:
                    key, value = line.strip().split("=", 1)
                    if key == "kinematics_path":
                        self.kinematics_path = value
                    elif key == "fluid_dynamics_path":
                        self.fluid_dynamics_path = value
                    elif key == "audio_waveform_path":
                        self.audio_waveform_path = value
                    elif key == "CAD_path":
                        self.CAD_path = value
                        self.cad_path_label.setText(os.path.basename(value))
                    elif key == "velocity":
                        self.velocity = value
                        self.velocity_input.setText(value)
                    elif key == "dark_mode":
                        self.dark_mode = value.lower() == "true"
                    elif key == "multithreading":
                        self.multithreading = value.lower() == "true"
                    elif key == "gpu_rendering":
                        self.gpu_rendering = value.lower() == "true"

    def save_kinematics_data(self):
        # Save the CSV file to the main project directory
        project_dir = os.getcwd()
        csv_path = os.path.join(project_dir, "kinematics_data.csv")

        angle_unit_val = UNIT_MAPPING.get(self.angle_unit.currentText())
        velocity_unit_val = UNIT_MAPPING.get(self.velocity_unit.currentText())
        gravity_unit_val = UNIT_MAPPING.get(self.gravity_unit.currentText())
        area_unit_val = UNIT_MAPPING.get(self.area_unit.currentText())
        height_unit_val = UNIT_MAPPING.get(self.height_unit.currentText())

        data = [
            ["initial_velocity", self.initial_velocity.text(), velocity_unit_val],
            ["launch_angle", self.launch_angle.text(), angle_unit_val],
            ["surface_area", self.surface_area.text(), area_unit_val],
            ["launch_height", self.initial_height.text(), height_unit_val],
            ["gravity_constant", self.gravity.text(), gravity_unit_val]
        ]

        with open(csv_path, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(["Parameter", "Value", "Unit"])
            writer.writerows(data)

        QMessageBox.information(self, "Success", "Kinematics data saved successfully.")
        self.run_kinematics_script()

    def save_and_execute_fluid_dynamics(self):
        self.velocity = self.velocity_input.text()

        # Save velocity and unit to a CSV file
        project_dir = os.getcwd()
        csv_path = os.path.join(project_dir, "fluid_velocity.csv")

        velocity_unit_val = UNIT_MAPPING.get(self.velocity_unit.currentText())

        with open(csv_path, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(["Velocity", "Unit"])
            writer.writerow([self.velocity, velocity_unit_val])

        self.save_settings()

        try:
            script_path = os.path.join(os.getcwd(), "fluid_dynamics.py")
            os.system(f'python "{script_path}"')
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to run fluid_dynamics.py: {e}")

    def run_kinematics_script(self):
        try:
            script_path = os.path.join(os.getcwd(), "kinematics.py")
            os.system(f'python "{script_path}"')
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to run kinematics.py: {e}")

    def clear_layout(self, layout):
        while layout.count():
            child = layout.takeAt(0)
            if child.widget():
                child.widget().deleteLater()

    def center_window(self):
        # Center the main window on the user's screen
        screen_geometry = QApplication.desktop().screenGeometry()
        screen_width = screen_geometry.width()
        screen_height = screen_geometry.height()
        x = int((screen_width - self.width()) / 2)
        y = int((screen_height - self.height()) / 2)
        self.move(x, y)

if __name__ == "__main__":
    app = QApplication(sys.argv)

    logo_path = os.path.join(os.getcwd(), "simucadsuitelogo.png")
    favicon_path = os.path.join(os.getcwd(), "simucadsuite_favicon_32x32.png")

    main_window = MainWindow(logo_path, favicon_path)
    main_window.hide()

    loading_screen = LoadingScreen(main_window, logo_path)
    QTimer.singleShot(8000, lambda: (loading_screen.close(), main_window.show()))  # Show main window with standard window decorations
    loading_screen.exec_()

    main_window.showMaximized()  # Ensure the window is maximized after it's shown

    sys.exit(app.exec_())
