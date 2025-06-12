import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load the image
image_path = 'practice wave.jpg'
img = cv2.imread(image_path)

if img is None:
    print(f"Error: Could not load image from {image_path}")
    exit(1)

# Convert to grayscale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Apply Gaussian blur to reduce noise
blurred = cv2.GaussianBlur(gray, (5, 5), 0)

# Apply edge detection
edges = cv2.Canny(blurred, 50, 150)

# Find the middle column of the image
middle_x = img.shape[1] // 2

# Extract the middle column of the edge-detected image
middle_column = edges[:, middle_x]

# Find the wave boundaries (top and bottom edges)
wave_points = np.where(middle_column > 0)[0]

if len(wave_points) < 2:
    print("Could not detect clear wave boundaries in the middle of the image")
    # Try with a broader area around the middle
    middle_area = edges[:, middle_x-10:middle_x+10]
    wave_points = np.where(np.any(middle_area > 0, axis=1))[0]

if len(wave_points) >= 2:
    # Calculate wave height in pixels
    wave_top = wave_points.min()
    wave_bottom = wave_points.max()
    wave_height_pixels = wave_bottom - wave_top
    print(f"Wave height in the middle: {wave_height_pixels} pixels")
    
    # Visualize the results
    result_img = img.copy()
    cv2.line(result_img, (middle_x, wave_top), (middle_x, wave_bottom), (0, 0, 255), 2)
    cv2.putText(result_img, f"Height: {wave_height_pixels}px", 
                (middle_x + 10, (wave_top + wave_bottom) // 2),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    
    # Convert from BGR to RGB for matplotlib
    result_img_rgb = cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB)
    
    # Display the results
    plt.figure(figsize=(12, 8))
    
    plt.subplot(221)
    plt.title('Original Image')
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    
    plt.subplot(222)
    plt.title('Edge Detection')
    plt.imshow(edges, cmap='gray')
    
    plt.subplot(223)
    plt.title('Wave Height Measurement')
    plt.imshow(result_img_rgb)
    
    plt.subplot(224)
    plt.title('Middle Column Profile')
    plt.plot(middle_column, range(len(middle_column)))
    plt.axhline(y=wave_top, color='r', linestyle='-', alpha=0.5)
    plt.axhline(y=wave_bottom, color='r', linestyle='-', alpha=0.5)
    plt.gca().invert_yaxis()  # Invert y-axis to match image coordinates
    
    plt.tight_layout()
    plt.savefig('wave_analysis_result.png')
    plt.show()
else:
    print("Could not detect wave boundaries in the image") 