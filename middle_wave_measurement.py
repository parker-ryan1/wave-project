import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load the image
image_path = 'practice wave.jpg'
img = cv2.imread(image_path)

if img is None:
    print(f"Error: Could not load image from {image_path}")
    exit(1)

# Get image dimensions
height, width = img.shape[:2]

# Define the region of interest (ROI) around the middle wave
# Focus on the middle portion of the image
roi_x_start = width // 3
roi_x_end = 2 * width // 3
roi_y_start = 0
roi_y_end = height
roi = img[roi_y_start:roi_y_end, roi_x_start:roi_x_end]

# Convert ROI to grayscale
gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

# Apply Gaussian blur to reduce noise
blurred_roi = cv2.GaussianBlur(gray_roi, (5, 5), 0)

# Apply adaptive thresholding to handle varying lighting conditions
binary_roi = cv2.adaptiveThreshold(blurred_roi, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                  cv2.THRESH_BINARY_INV, 11, 2)

# Apply morphological operations to clean up the binary image
kernel = np.ones((5, 5), np.uint8)
opening = cv2.morphologyEx(binary_roi, cv2.MORPH_OPEN, kernel, iterations=2)
closing = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, kernel, iterations=2)

# Find contours in the binary image
contours, _ = cv2.findContours(closing, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Filter contours by size to find the wave
min_contour_area = 1000  # Adjust based on your image
wave_contours = [cnt for cnt in contours if cv2.contourArea(cnt) > min_contour_area]

# If we found potential wave contours
if wave_contours:
    # Sort contours by area (largest first)
    wave_contours = sorted(wave_contours, key=cv2.contourArea, reverse=True)
    
    # Take the largest contour as the main wave
    main_wave_contour = wave_contours[0]
    
    # Get the bounding rectangle
    x, y, w, h = cv2.boundingRect(main_wave_contour)
    
    # Adjust coordinates back to the original image
    x_orig = x + roi_x_start
    y_orig = y + roi_y_start
    
    # Draw the rectangle on the original image
    result_img = img.copy()
    cv2.rectangle(result_img, (x_orig, y_orig), (x_orig+w, y_orig+h), (0, 255, 0), 2)
    
    # Calculate the height in pixels
    height_pixels = h
    
    # Estimate conversion from pixels to centimeters
    # This is an approximation - adjust based on your knowledge of the wave
    # For a typical ocean wave in this type of photo, we can estimate:
    # Small wave: ~30-60 cm
    # Medium wave: ~60-150 cm
    # Large wave: ~150-300+ cm
    
    # Let's assume this is a medium-sized wave of around 100 cm
    estimated_height_cm = 100
    pixel_to_cm_ratio = estimated_height_cm / height_pixels
    
    # Calculate height in centimeters
    height_cm = height_pixels * pixel_to_cm_ratio
    
    print(f"Middle wave bounding box height: {height_pixels} pixels")
    print(f"Estimated wave height: {height_cm:.1f} cm")
    print(f"(Based on an estimated scale of {pixel_to_cm_ratio:.4f} cm/pixel)")
    
    # Add text with measurements to the image
    cv2.putText(result_img, f"Height: {height_pixels} px", (x_orig+w+10, y_orig+h//2-20),
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    cv2.putText(result_img, f"~{height_cm:.1f} cm", (x_orig+w+10, y_orig+h//2+20),
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    
    # Create a visualization to show the process
    # Create a contour image for visualization
    contour_img = np.zeros_like(gray_roi)
    cv2.drawContours(contour_img, [main_wave_contour], -1, 255, 2)
    
    # Display the results
    plt.figure(figsize=(15, 10))
    
    plt.subplot(231)
    plt.title('Original Image')
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    
    plt.subplot(232)
    plt.title('ROI (Middle Section)')
    plt.imshow(cv2.cvtColor(roi, cv2.COLOR_BGR2RGB))
    
    plt.subplot(233)
    plt.title('Binary Image')
    plt.imshow(closing, cmap='gray')
    
    plt.subplot(234)
    plt.title('Detected Wave Contour')
    plt.imshow(contour_img, cmap='gray')
    
    plt.subplot(235)
    plt.title('Wave Height Measurement')
    plt.imshow(cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB))
    
    # Add a scale reference
    plt.subplot(236)
    plt.title('Scale Reference')
    # Create a blank image for the scale
    scale_img = np.ones((300, 300, 3), dtype=np.uint8) * 255
    
    # Draw a 100 cm reference line
    pixels_for_100cm = int(100 / pixel_to_cm_ratio)
    start_y = 150
    start_x = 50
    end_x = start_x + pixels_for_100cm
    
    cv2.line(scale_img, (start_x, start_y), (end_x, start_y), (0, 0, 0), 2)
    cv2.line(scale_img, (start_x, start_y-5), (start_x, start_y+5), (0, 0, 0), 2)
    cv2.line(scale_img, (end_x, start_y-5), (end_x, start_y+5), (0, 0, 0), 2)
    cv2.putText(scale_img, "100 cm", (start_x + pixels_for_100cm//2 - 30, start_y - 20),
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
    
    plt.imshow(scale_img)
    
    plt.tight_layout()
    plt.savefig('middle_wave_measurement_result.png')
    plt.show()
else:
    print("Could not detect the middle wave in the image") 