# SimuCADSuite

![alt text](https://github.com/JoshBaneyCS/SimuCADSuite/blob/main/simucadsuitelogo.png?raw=true)

SimuCADSuite is a comprehensive, modular simulation tool designed as an open-source physics simulation tool. The Program is made to fulfill the Undergraduate capstone for a BS- CS degree and to highlight scientific computing/ mathmatical analysis and computer science interests of the author. It combines the power of Python for complex computational tasks with the flexibility of Electron for a modern, interactive user interface. The suite includes modules for Fluid Dynamics Simulation, Kinematics Simulation, and Audio Decoding using Fast Fourier Transform and Discrete Fourier Transform (FFT and DFT), making it a versatile tool for a variety of engineering and scientific applications.

## Table of Contents

- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
  - [Home Window](#home-window)
  - [Fluid Dynamics Simulation](#fluid-dynamics-simulation)
  - [Kinematics Simulation](#kinematics-simulation)
  - [Audio Decoding](#audio-decoding)
- [System Requirements](#system-requirements)
- [Contributing](#contributing)
- [License](#license)

## Features

- **Fluid Dynamics Simulation**: Simulate the flow of particles around CAD models using advanced fluid dynamics equations like Navier-Stokes, Euler’s equations, and Bernoulli's principle. Convert CAD models to NumPy arrays for simulation and visualize the results with vector fields.
  
- **Kinematics Simulation**: Analyze projectile motion with and without air resistance, visualize the paths and vector fields, and compare results in real-time. User-defined inputs for velocity, angle, drag, and more.
  

- **Audio Decoding**: Decompose audio files into their constituent sine waves using FFT/DFT, visualize frequency spectra, and analyze the results with user-defined parameters.

- **Interactive GUI**: A modern Electron-based interface allows seamless navigation between modules, with an intuitive home window and back buttons for easy access to different tools.

- **Multi-threading and GPU Acceleration**: Select between single-threaded, multi-threaded, or GPU-accelerated computation for optimal performance.

- **Data Management**: Save and load simulation results in various formats, including Excel, CSV, and SQL databases.

## Installation

To get started with SimuCADSuite, follow these steps:

### Prerequisites

Ensure you have the following installed on your system:

- **Python 3.8+**
- **Node.js 14+**
- **Git**

### Clone the Repository

```bash
git clone https://github.com/JoshBaneyCS/SimuCADSuite.git
cd SimuCADSuite
