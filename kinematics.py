import csv
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PyQt5.QtWidgets import QApplication, QMainWindow, QPushButton, QFileDialog, QMessageBox, QWidget, QVBoxLayout, \
    QComboBox, QLabel, QSpacerItem, QSizePolicy, QLineEdit, QGridLayout, QScrollArea
from PyQt5.QtGui import QFont, QIcon
import sys

# Unit Mapping
UNIT_MAPPING = {
    "m/s": 1,
    "ft/s": 2,
    "mph": 3,
    "km/h": 4,
    "m": 5,
    "ft": 6,
    "cm": 7,
    "in": 8,
    "m²": 9,
    "ft²": 10,
    "cm²": 11,
    "in²": 12,
    "Degrees": 13,
    "Radians": 14,
    "m/s²": 15,
    "ft/s²": 16,
    "g": 17
}

# Drag Coefficients for different shapes
DRAG_COEFFICIENTS = {
    "Circle": 1.17,
    "Square": 2.1,
    "Rhombus": 1.6,
    "Airfoil": 0.045,
    "Sphere": 0.47
}

# Mass Conversion Factors
MASS_CONVERSION = {
    "g": 1e-3,  # grams to kilograms
    "kg": 1.0,  # kilograms to kilograms
    "lbs": 0.453592,  # pounds to kilograms
    "tons": 1000.0  # tons to kilograms
}

# Velocity Conversion Factors
VELOCITY_CONVERSION = {
    "m/s": 1.0,
    "ft/s": 3.28084,
    "mph": 2.23694,
    "km/h": 3.6
}

# Distance Conversion Factors
DISTANCE_CONVERSION = {
    "m": 1.0,
    "ft": 3.28084,
    "cm": 100.0,
    "in": 39.3701
}


# Function to convert units
def convert_units(value, unit_type):
    # Velocity units
    if unit_type == 1:  # m/s
        return value  # Already in m/s
    elif unit_type == 2:  # ft/s
        return value * 0.3048  # Convert to m/s
    elif unit_type == 3:  # mph
        return value * 0.44704  # Convert to m/s
    elif unit_type == 4:  # km/h
        return value * 0.277778  # Convert to m/s

    # Distance units
    elif unit_type == 5:  # m
        return value  # Already in meters
    elif unit_type == 6:  # ft
        return value * 0.3048  # Convert to meters
    elif unit_type == 7:  # cm
        return value * 0.01  # Convert to meters
    elif unit_type == 8:  # in
        return value * 0.0254  # Convert to meters

    # Area units
    elif unit_type == 9:  # m²
        return value  # Already in square meters
    elif unit_type == 10:  # ft²
        return value * 0.092903  # Convert to square meters
    elif unit_type == 11:  # cm²
        return value * 0.0001  # Convert to square meters
    elif unit_type == 12:  # in²
        return value * 0.00064516  # Convert to square meters

    # Angle units
    elif unit_type == 13:  # Degrees
        return math.radians(value)  # Convert to radians
    elif unit_type == 14:  # Radians
        return value  # Already in radians

    # Acceleration units
    elif unit_type == 15:  # m/s²
        return value  # Already in m/s²
    elif unit_type == 16:  # ft/s²
        return value * 0.3048  # Convert to m/s²
    elif unit_type == 17:  # g (acceleration due to gravity)
        return value * 9.80665  # Convert to m/s²

    else:
        raise ValueError(f"Unsupported unit type: {unit_type}")


# Function to load and convert kinematics data from CSV
def load_data_from_csv(filename="kinematics_data.csv"):
    data = {}

    # Read the CSV file
    with open(filename, mode='r') as file:
        reader = csv.DictReader(file)
        for row in reader:
            param = row['Parameter']
            value = float(row['Value'])
            unit_str = row['Unit'].strip()

            # Set default units if missing
            default_units = {
                "initial_velocity": 1,  # Default to m/s
                "launch_angle": 13,  # Default to Degrees
                "surface_area": 9,  # Default to m²
                "launch_height": 5,  # Default to meters
                "gravity_constant": 15  # Default to m/s²
            }

            if unit_str.isdigit():
                unit = int(unit_str)
            else:
                unit = default_units.get(param)
                if unit is None:
                    print(f"Warning: Invalid or missing unit for parameter '{param}', skipping this entry.")
                    continue

            data[param] = convert_units(value, unit)

    # Extract the necessary parameters
    initial_velocity = data.get("initial_velocity", 0.0)
    launch_angle = data.get("launch_angle", 0.0)
    surface_area = data.get("surface_area", 0.1)
    launch_height = data.get("launch_height", 0.0)
    gravity_constant = data.get("gravity_constant", 9.81)

    return initial_velocity, launch_angle, gravity_constant, surface_area, launch_height


# Function to calculate trajectory without air resistance
def calculate_trajectory_without_air_resistance(v0, angle, g, h0=0):
    t_flight = (v0 * math.sin(angle) + math.sqrt((v0 * math.sin(angle)) ** 2 + 2 * g * h0)) / g
    t = np.linspace(0, t_flight, num=1000)
    x = v0 * np.cos(angle) * t
    y = h0 + v0 * np.sin(angle) * t - 0.5 * g * t ** 2
    return t, x, y, max(y)


# Function to calculate trajectory with air resistance using drag equation
def calculate_trajectory_with_drag(v0, angle, g, Cd, A, mass, h0=0):
    rho = 1.225  # Air density (kg/m^3)
    dt = 0.01  # Time step (s)

    t = np.arange(0, 10, dt)
    vx = np.zeros_like(t)
    vy = np.zeros_like(t)
    x = np.zeros_like(t)
    y = np.zeros_like(t)

    vx[0] = v0 * np.cos(angle)
    vy[0] = v0 * np.sin(angle)
    x[0] = 0
    y[0] = h0

    for i in range(1, len(t)):
        v = np.sqrt(vx[i - 1] ** 2 + vy[i - 1] ** 2)
        Fd = 0.5 * Cd * A * rho * v ** 2

        ax = -Fd * vx[i - 1] / mass
        ay = -g - (Fd * vy[i - 1] / mass)

        vx[i] = vx[i - 1] + ax * dt
        vy[i] = vy[i - 1] + ay * dt

        x[i] = x[i - 1] + vx[i] * dt
        y[i] = y[i - 1] + vy[i] * dt

        if y[i] < 0:
            break

    return t[:i + 1], x[:i + 1], y[:i + 1], max(y[:i + 1])


# Function to generate random points along the parabola and calculate velocities
def generate_random_points_with_vectors(t, x, y, v0, angle, g):
    random_indices = np.linspace(0, len(t) - 1, 20, dtype=int)  # Sample 20 points along the trajectory
    points = []
    for i in random_indices:
        vx = v0 * np.cos(angle)
        vy = v0 * np.sin(angle) - g * t[i]
        vt = np.sqrt(vx ** 2 + vy ** 2)
        points.append((x[i], y[i], vx, vy, vt, t[i]))
    return points


# Function to save data to Excel
def save_to_excel(data, filename):
    df = pd.DataFrame(data, columns=['x', 'y', 'vx', 'vy', 'vt', 'time'])
    df.to_excel(filename, index=False)


# Plotting function with smaller and less cluttered vectors in a resizable window
def plot_trajectories_with_vectors(t1, x1, y1, t2, x2, y2, points, v0, angle, g):
    # Create a resizable plot window
    plt.figure(figsize=(12, 8))

    plt.plot(x1, y1, label="Without Air Resistance", linestyle='--')
    plt.plot(x2, y2, label="With Air Resistance", linestyle='-')

    # Plot smaller, less cluttered velocity vectors
    for i, point in enumerate(points):
        if i % 2 == 0:  # Only plot every second point to reduce clutter
            plt.quiver(point[0], point[1], point[2], point[3], angles='xy', scale_units='xy', scale=50, color='r')
            plt.quiver(point[0], point[1], point[2], 0, angles='xy', scale_units='xy', scale=50, color='g',
                       alpha=0.5)  # vx
            plt.quiver(point[0], point[1], 0, point[3], angles='xy', scale_units='xy', scale=50, color='b',
                       alpha=0.5)  # vy

    plt.xlabel("Distance (m)")
    plt.ylabel("Height (m)")
    plt.legend()
    plt.title("Projectile Motion with and without Drag")
    plt.grid(True)
    plt.show()


# PyQt5 GUI Setup
class KinematicsApp(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("Projectile Motion Simulator")
        self.setWindowIcon(QIcon("simucadsuite_favicon_32x32.png"))
        self.setGeometry(100, 100, 800, 600)
        self.showMaximized()

        # Central widget and layout with scrolling
        self.central_widget = QWidget(self)
        self.scroll_area = QScrollArea(self)
        self.scroll_area.setWidgetResizable(True)
        self.scroll_area.setWidget(self.central_widget)
        self.setCentralWidget(self.scroll_area)

        self.layout = QVBoxLayout(self.central_widget)

        # Shape selection combo box
        self.shape_label = QLabel("Select Shape:", self)
        self.shape_label.setFont(QFont("Arial", 14))
        self.layout.addWidget(self.shape_label)

        self.shape_combo_box = QComboBox(self)
        self.shape_combo_box.setFont(QFont("Arial", 14))
        self.shape_combo_box.addItems(DRAG_COEFFICIENTS.keys())
        self.layout.addWidget(self.shape_combo_box)

        # Mass input
        self.mass_label = QLabel("Enter Mass:", self)
        self.mass_label.setFont(QFont("Arial", 14))
        self.layout.addWidget(self.mass_label)

        self.mass_input = QLineEdit(self)
        self.mass_input.setFont(QFont("Arial", 14))
        self.layout.addWidget(self.mass_input)

        self.mass_unit_combo_box = QComboBox(self)
        self.mass_unit_combo_box.setFont(QFont("Arial", 14))
        self.mass_unit_combo_box.addItems(MASS_CONVERSION.keys())
        self.layout.addWidget(self.mass_unit_combo_box)

        # Add spacer for better aesthetics
        self.layout.addSpacerItem(QSpacerItem(20, 40, QSizePolicy.Minimum, QSizePolicy.Expanding))

        # Load data button
        self.load_button = QPushButton("Load Data", self)
        self.load_button.setFont(QFont("Arial", 14))
        self.load_button.setFixedHeight(50)
        self.layout.addWidget(self.load_button)
        self.load_button.clicked.connect(self.load_data)

        # Calculate button
        self.calculate_button = QPushButton("Calculate", self)
        self.calculate_button.setFont(QFont("Arial", 14))
        self.calculate_button.setFixedHeight(50)
        self.layout.addWidget(self.calculate_button)
        self.calculate_button.clicked.connect(self.calculate_trajectories)

        # Save to Excel button
        self.save_button = QPushButton("Save to Excel", self)
        self.save_button.setFont(QFont("Arial", 14))
        self.save_button.setFixedHeight(50)
        self.layout.addWidget(self.save_button)
        self.save_button.clicked.connect(self.save_data)

        # Data output labels
        self.data_grid_layout = QGridLayout()
        self.layout.addLayout(self.data_grid_layout)

        self.data_labels = {}
        self.data_combo_boxes = {}

        self.setup_data_output()

    def setup_data_output(self):
        # For point velocities and elapsed times (20 points)
        for i in range(20):
            label = QLabel(f"Point {i + 1}:", self)
            label.setFont(QFont("Arial", 14))
            self.data_grid_layout.addWidget(label, i, 0)

            self.data_labels[f"Point {i + 1}"] = QLabel("Vx= N/A, Vy= N/A, Vtangent= N/A, Elapsed time= N/A", self)
            self.data_labels[f"Point {i + 1}"].setFont(QFont("Arial", 14))
            self.data_grid_layout.addWidget(self.data_labels[f"Point {i + 1}"], i, 1)

            combo_box = QComboBox(self)
            combo_box.setFont(QFont("Arial", 14))
            combo_box.addItems(VELOCITY_CONVERSION.keys())
            combo_box.currentIndexChanged.connect(lambda _, p=i: self.update_point_velocity_units(p))
            self.data_grid_layout.addWidget(combo_box, i, 2)
            self.data_combo_boxes[f"Point {i + 1}"] = combo_box

        # For Max Height, Distance Traveled, and Total Time
        additional_params = ["Max Height", "Distance Traveled", "Total Time"]
        for i, param in enumerate(additional_params, start=20):
            label = QLabel(f"{param}: ", self)
            label.setFont(QFont("Arial", 14))
            self.data_grid_layout.addWidget(label, i, 0)
            self.data_labels[param] = QLabel("N/A", self)
            self.data_labels[param].setFont(QFont("Arial", 14))
            self.data_grid_layout.addWidget(self.data_labels[param], i, 1)

            if param != "Total Time":
                combo_box = QComboBox(self)
                combo_box.setFont(QFont("Arial", 14))
                combo_box.addItems(DISTANCE_CONVERSION.keys())
                combo_box.currentIndexChanged.connect(lambda _, p=param: self.update_output_units(p))
                self.data_grid_layout.addWidget(combo_box, i, 2)
                self.data_combo_boxes[param] = combo_box

    def update_point_velocity_units(self, point_index):
        point_key = f"Point {point_index + 1}"
        conversion_factor = VELOCITY_CONVERSION[self.data_combo_boxes[point_key].currentText()]
        point_data = self.points[point_index]
        vx, vy, vt, elapsed_time = point_data[2] * conversion_factor, point_data[3] * conversion_factor, point_data[
            4] * conversion_factor, point_data[5]
        self.data_labels[point_key].setText(
            f"Vx= {vx:.2f} {self.data_combo_boxes[point_key].currentText()}, Vy= {vy:.2f} {self.data_combo_boxes[point_key].currentText()}, Vtangent= {vt:.2f} {self.data_combo_boxes[point_key].currentText()}, Elapsed time= {elapsed_time:.2f} s")

    def update_output_units(self, param):
        if param in ["Max Height", "Distance Traveled"]:
            conversion_factor = DISTANCE_CONVERSION[self.data_combo_boxes[param].currentText()]
            self.data_labels[param].setText(
                f"{self.data_values[param] * conversion_factor:.2f} {self.data_combo_boxes[param].currentText()}")
        elif param == "Total Time":
            self.data_labels[param].setText(f"{self.data_values[param]:.2f} s")

    def load_data(self):
        self.data = load_data_from_csv()
        QMessageBox.information(self, "Data Loaded", f"Data loaded: {self.data}")

    def calculate_trajectories(self):
        # Ensure the mass input is valid
        mass_value_str = self.mass_input.text().strip()
        if not mass_value_str:
            QMessageBox.warning(self, "Input Error", "Please enter a valid mass.")
            return

        try:
            mass_value = float(mass_value_str)
        except ValueError:
            QMessageBox.warning(self, "Input Error", "Mass must be a numeric value.")
            return

        mass_unit = self.mass_unit_combo_box.currentText()
        mass = mass_value * MASS_CONVERSION[mass_unit]

        v0, angle, g, surface_area, h0 = self.data
        selected_shape = self.shape_combo_box.currentText()
        Cd = DRAG_COEFFICIENTS[selected_shape]

        t1, x1, y1, max_height_1 = calculate_trajectory_without_air_resistance(v0, angle, g, h0)
        t2, x2, y2, max_height_2 = calculate_trajectory_with_drag(v0, angle, g, Cd, surface_area, mass, h0)
        self.points = generate_random_points_with_vectors(t2, x2, y2, v0, angle, g)
        self.data_values = {
            "Max Height": max_height_2,
            "Distance Traveled": x2[-1],
            "Total Time": t2[-1]
        }
        for i in range(20):
            self.update_point_velocity_units(i)
        self.update_output_units("Max Height")
        self.update_output_units("Distance Traveled")
        self.update_output_units("Total Time")

        plot_trajectories_with_vectors(t1, x1, y1, t2, x2, y2, self.points, v0, angle, g)

    def save_data(self):
        filename, _ = QFileDialog.getSaveFileName(self, "Save to Excel", "", "Excel Files (*.xlsx)")
        if filename:
            save_to_excel(self.points, filename)
            QMessageBox.information(self, "Data Saved", f"Data saved to {filename}")


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = KinematicsApp()
    window.show()
    sys.exit(app.exec_())

