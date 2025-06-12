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

# Apply threshold to create a binary image
_, binary = cv2.threshold(blurred, 100, 255, cv2.THRESH_BINARY_INV)

# Find contours
contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Find the largest contour (assuming it's the wave)
if contours:
    largest_contour = max(contours, key=cv2.contourArea)
    
    # Get the bounding rectangle of the contour
    x, y, w, h = cv2.boundingRect(largest_contour)
    
    # Find the middle of the image
    middle_x = img.shape[1] // 2
    
    # Check if the middle point is inside the contour's x-range
    if x <= middle_x <= x + w:
        # Find all points of the contour
        contour_points = largest_contour.reshape(-1, 2)
        
        # Filter points near the middle x-coordinate
        tolerance = 10  # Pixels around the middle to consider
        middle_points = contour_points[np.abs(contour_points[:, 0] - middle_x) <= tolerance]
        
        if len(middle_points) > 0:
            # Find top and bottom points in the middle region
            top_y = np.min(middle_points[:, 1])
            bottom_y = np.max(middle_points[:, 1])
            
            # Calculate wave height
            wave_height_pixels = bottom_y - top_y
            print(f"Wave height in the middle: {wave_height_pixels} pixels")
            
            # Draw the contour and height measurement
            result_img = img.copy()
            cv2.drawContours(result_img, [largest_contour], -1, (0, 255, 0), 2)
            cv2.line(result_img, (middle_x, int(top_y)), (middle_x, int(bottom_y)), (0, 0, 255), 2)
            cv2.putText(result_img, f"Height: {wave_height_pixels}px", 
                       (middle_x + 10, int((top_y + bottom_y) // 2)),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            
            # Display the results
            plt.figure(figsize=(15, 10))
            
            plt.subplot(231)
            plt.title('Original Image')
            plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            
            plt.subplot(232)
            plt.title('Grayscale')
            plt.imshow(gray, cmap='gray')
            
            plt.subplot(233)
            plt.title('Binary')
            plt.imshow(binary, cmap='gray')
            
            plt.subplot(234)
            plt.title('Detected Contour')
            contour_img = np.zeros_like(gray)
            cv2.drawContours(contour_img, [largest_contour], -1, 255, 2)
            plt.imshow(contour_img, cmap='gray')
            
            plt.subplot(235)
            plt.title('Wave Height Measurement')
            plt.imshow(cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB))
            
            # Plot the contour points
            plt.subplot(236)
            plt.title('Contour Points')
            plt.scatter(contour_points[:, 0], contour_points[:, 1], s=1, c='blue')
            plt.scatter(middle_points[:, 0], middle_points[:, 1], s=3, c='red')
            plt.axvline(x=middle_x, color='g', linestyle='-', alpha=0.5)
            plt.axhline(y=top_y, color='r', linestyle='-', alpha=0.5)
            plt.axhline(y=bottom_y, color='r', linestyle='-', alpha=0.5)
            plt.gca().invert_yaxis()  # Invert y-axis to match image coordinates
            
            plt.tight_layout()
            plt.savefig('wave_analysis_contour_result.png')
            plt.show()
        else:
            print("No contour points found near the middle of the image")
    else:
        print("The largest contour does not intersect with the middle of the image")
else:
    print("No contours detected in the image") 