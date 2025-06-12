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

# Apply adaptive thresholding to better identify the wave
thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                              cv2.THRESH_BINARY, 11, 2)

# Invert the image so the wave appears as white on black
thresh_inv = cv2.bitwise_not(thresh)

# Apply morphological operations to clean up the image
kernel = np.ones((3, 3), np.uint8)
opening = cv2.morphologyEx(thresh_inv, cv2.MORPH_OPEN, kernel, iterations=2)
closing = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, kernel, iterations=2)

# Find the middle column of the image
middle_x = img.shape[1] // 2

# Get a range of columns around the middle for more robust detection
column_range = 20
middle_region = closing[:, middle_x-column_range:middle_x+column_range]
middle_profile = np.max(middle_region, axis=1)

# Find the wave boundaries
wave_points = np.where(middle_profile > 0)[0]

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
    plt.figure(figsize=(15, 10))
    
    plt.subplot(231)
    plt.title('Original Image')
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    
    plt.subplot(232)
    plt.title('Grayscale')
    plt.imshow(gray, cmap='gray')
    
    plt.subplot(233)
    plt.title('Thresholded')
    plt.imshow(thresh_inv, cmap='gray')
    
    plt.subplot(234)
    plt.title('Morphological Processing')
    plt.imshow(closing, cmap='gray')
    
    plt.subplot(235)
    plt.title('Wave Height Measurement')
    plt.imshow(result_img_rgb)
    
    plt.subplot(236)
    plt.title('Middle Region Profile')
    plt.plot(middle_profile, range(len(middle_profile)))
    plt.axhline(y=wave_top, color='r', linestyle='-', alpha=0.5)
    plt.axhline(y=wave_bottom, color='r', linestyle='-', alpha=0.5)
    plt.gca().invert_yaxis()  # Invert y-axis to match image coordinates
    
    plt.tight_layout()
    plt.savefig('wave_analysis_alt_result.png')
    plt.show()
else:
    print("Could not detect wave boundaries in the image") 