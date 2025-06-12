# Simulated Camera Project


READ THIS IS FROM ME PARKER:
$
    Dont forge to change the dir path in the code
    A table Simulated Images is where the images will be stored
    fix this part: private static string connectionString = "Server=YOUR_SERVER;Database=YOUR_DATABASE;User Id=YOUR_USERNAME;Password=YOUR_PASSWORD;TrustServerCertificate=True;";
$

## Overview
This project simulates a camera system that takes pictures at regular intervals and saves them to a specified directory. It's designed for testing and development purposes where a physical camera is not available.

## Features
- Simulates taking pictures at configurable intervals
- Saves images with timestamp-based filenames
- Processes images and generates result files
- Configurable number of pictures per sequence
- Configurable time interval between sequences

## Configuration
CHANGE THESE PARKER SO IT WORKS FOR YOU
The program has the following default settings:
- Number of pictures per sequence: 50
- Time interval between sequences: 2 minutes
- Output directory: `C:\Users\rdupart\OneDrive - Laborde Products Inc\camera\SimulatedImages`

## Requirements
- .NET 9.0 or later
- Windows operating system

## Building and Running
1. Clone the repository
2. Open the solution in Visual Studio or your preferred IDE
3. Build the solution
4. Run the program

## Output
The program generates:
- Simulated image files with format: `Picture_YYYYMMDD_HHMMSS_N.jpg`
- Result files with format: `Picture_YYYYMMDD_HHMMSS_N_result.txt`

## Directory Structure
```
camera/
├── SimulatedCamera/         # Main project directory
│   ├── Program.cs          # Main program logic
│   └── SimulatedCamera.csproj
└── SimulatedImages/        # Output directory for generated images
```

## Notes
- The program runs continuously until manually stopped
- Each sequence of pictures is followed by a waiting period
- The output directory is created automatically if it doesn't exist

## License
This project is proprietary and confidential. All rights reserved. 