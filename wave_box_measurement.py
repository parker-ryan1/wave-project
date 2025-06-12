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
# Adjust threshold values as needed for your specific image
_, binary = cv2.threshold(blurred, 100, 255, cv2.THRESH_BINARY_INV)

# Apply morphological operations to clean up the binary image
kernel = np.ones((5, 5), np.uint8)
binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

# Find contours in the binary image
contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Find the contour in the middle of the image (assuming it's the wave we're interested in)
middle_x = img.shape[1] // 2
middle_y = img.shape[0] // 2

# Find the contour closest to the middle of the image
target_contour = None
min_distance = float('inf')

for contour in contours:
    # Calculate the center of the contour
    M = cv2.moments(contour)
    if M["m00"] != 0:
        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])
        
        # Calculate distance from the middle
        distance = np.sqrt((cx - middle_x)**2 + (cy - middle_y)**2)
        
        # Check if this is the closest contour so far
        if distance < min_distance:
            min_distance = distance
            target_contour = contour

# If we found a contour
if target_contour is not None:
    # Get the bounding rectangle
    x, y, w, h = cv2.boundingRect(target_contour)
    
    # Draw the rectangle on the image
    result_img = img.copy()
    cv2.rectangle(result_img, (x, y), (x+w, y+h), (0, 255, 0), 2)
    
    # Calculate the height in pixels
    height_pixels = h
    
    # Estimate conversion from pixels to centimeters
    # This is an approximation - for accurate measurements, you would need a reference object of known size in the image
    # Assuming average wave height of around 60-100 cm for this type of wave
    # You can adjust this conversion factor based on your knowledge of the actual wave size
    pixel_to_cm_ratio = 100 / height_pixels  # Estimate: the wave height is approximately 100 cm
    
    height_cm = height_pixels * pixel_to_cm_ratio
    
    print(f"Wave bounding box height: {height_pixels} pixels")
    print(f"Estimated wave height: {height_cm:.1f} cm (using estimated scale)")
    
    # Add text with measurements to the image
    cv2.putText(result_img, f"Height: {height_pixels} px", (x+w+10, y+h//2-20),
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    cv2.putText(result_img, f"~{height_cm:.1f} cm", (x+w+10, y+h//2+20),
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    
    # Display the results
    plt.figure(figsize=(12, 8))
    
    plt.subplot(221)
    plt.title('Original Image')
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    
    plt.subplot(222)
    plt.title('Binary Image')
    plt.imshow(binary, cmap='gray')
    
    plt.subplot(223)
    plt.title('Contours')
    contour_img = np.zeros_like(gray)
    cv2.drawContours(contour_img, [target_contour], -1, 255, 2)
    plt.imshow(contour_img, cmap='gray')
    
    plt.subplot(224)
    plt.title('Wave Height Measurement')
    plt.imshow(cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB))
    
    plt.tight_layout()
    plt.savefig('wave_box_measurement_result.png')
    plt.show()
else:
    print("Could not find a suitable contour for the wave") 