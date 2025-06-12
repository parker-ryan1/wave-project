import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load the image
image_path = 'practice wave.jpg'
img = cv2.imread(image_path)

if img is None:
    print(f"Error: Could not load image from {image_path}")
    exit(1)

# Convert to HSV color space for better color segmentation
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

# Define range for blue/green water color (adjust these values based on your image)
# Lower bound - darker blue/green water
lower_water = np.array([90, 50, 50])
# Upper bound - lighter blue/green water
upper_water = np.array([130, 255, 255])

# Create a mask for water pixels
water_mask = cv2.inRange(hsv, lower_water, upper_water)

# Apply morphological operations to clean up the mask
kernel = np.ones((5, 5), np.uint8)
water_mask = cv2.morphologyEx(water_mask, cv2.MORPH_OPEN, kernel)
water_mask = cv2.morphologyEx(water_mask, cv2.MORPH_CLOSE, kernel)

# Find contours in the water mask
contours, _ = cv2.findContours(water_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Find the largest contour (assumed to be the water)
if contours:
    water_contour = max(contours, key=cv2.contourArea)
    
    # Create a mask with just the largest water contour
    water_area = np.zeros_like(water_mask)
    cv2.drawContours(water_area, [water_contour], -1, 255, -1)
    
    # Find the middle column of the image
    middle_x = img.shape[1] // 2
    
    # Get the water surface line by scanning from top to bottom in the middle column
    water_surface_y = 0
    for y in range(img.shape[0]):
        if water_area[y, middle_x] > 0:
            water_surface_y = y
            break
    
    # Apply edge detection to find the wave peaks
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 50, 150)
    
    # Create a region of interest around the middle
    roi_width = 40  # pixels on each side of the middle
    roi = edges[:, middle_x-roi_width:middle_x+roi_width]
    
    # Sum across the ROI width to get a stronger signal
    edge_profile = np.sum(roi, axis=1)
    
    # Find significant edges below the water surface
    wave_peaks = []
    for i in range(water_surface_y + 10, len(edge_profile) - 1):  # Start a bit below the surface
        # Look for peaks in the edge profile
        if edge_profile[i] > 50 and edge_profile[i] > edge_profile[i-1] and edge_profile[i] > edge_profile[i+1]:
            wave_peaks.append(i)
    
    if wave_peaks:
        # Take the first peak as the middle wave peak
        wave_peak_y = wave_peaks[0]
        
        # Calculate wave height
        wave_height_pixels = wave_peak_y - water_surface_y
        print(f"Estimated wave height from surface: {wave_height_pixels} pixels")
        
        # Visualize the results
        result_img = img.copy()
        
        # Draw water surface line
        cv2.line(result_img, (middle_x - 100, water_surface_y), (middle_x + 100, water_surface_y), (0, 255, 0), 2)
        cv2.putText(result_img, "Water Surface", (middle_x - 200, water_surface_y - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Draw wave peak and height measurement
        cv2.line(result_img, (middle_x, water_surface_y), (middle_x, wave_peak_y), (0, 0, 255), 2)
        cv2.putText(result_img, f"Wave Height: {wave_height_pixels}px", 
                   (middle_x + 10, (water_surface_y + wave_peak_y) // 2),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        # Display the results
        plt.figure(figsize=(15, 10))
        
        plt.subplot(231)
        plt.title('Original Image')
        plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        
        plt.subplot(232)
        plt.title('Water Mask')
        plt.imshow(water_mask, cmap='gray')
        
        plt.subplot(233)
        plt.title('Water Area')
        plt.imshow(water_area, cmap='gray')
        
        plt.subplot(234)
        plt.title('Edge Detection')
        plt.imshow(edges, cmap='gray')
        
        plt.subplot(235)
        plt.title('Wave Height Measurement')
        plt.imshow(cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB))
        
        plt.subplot(236)
        plt.title('Edge Profile')
        plt.plot(edge_profile, range(len(edge_profile)))
        plt.axhline(y=water_surface_y, color='g', linestyle='-', alpha=0.5)
        plt.axhline(y=wave_peak_y, color='r', linestyle='-', alpha=0.5)
        for peak in wave_peaks:
            plt.axhline(y=peak, color='b', linestyle='--', alpha=0.3)
        plt.gca().invert_yaxis()  # Invert y-axis to match image coordinates
        
        plt.tight_layout()
        plt.savefig('wave_height_color_segmentation_result.png')
        plt.show()
    else:
        print("Could not detect wave peaks below the water surface")
else:
    print("Could not detect water in the image") 