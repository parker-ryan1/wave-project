import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import RectangleSelector

# Global variables
img = None
ax = None
fig = None
wave_box = None  # Will store [x, y, width, height]
pixel_to_cm_ratio = 1.0  # Default ratio

def line_select_callback(eclick, erelease):
    """
    Callback for line selection.
    
    Parameters:
    eclick and erelease - the press and release events
    """
    global wave_box
    x1, y1 = eclick.xdata, eclick.ydata
    x2, y2 = erelease.xdata, erelease.ydata
    wave_box = [min(x1, x2), min(y1, y2), abs(x2-x1), abs(y2-y1)]
    
    # Update the plot with the selected box
    update_plot()

def update_plot():
    """Update the plot with the current wave_box."""
    global img, ax, fig, wave_box
    
    if wave_box is None:
        return
    
    # Clear the current axis
    ax.clear()
    
    # Display the image
    ax.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    
    # Draw the rectangle
    x, y, w, h = wave_box
    rect = plt.Rectangle((x, y), w, h, fill=False, edgecolor='green', linewidth=2)
    ax.add_patch(rect)
    
    # Calculate height in centimeters
    height_pixels = h
    height_cm = height_pixels * pixel_to_cm_ratio
    
    # Add text with measurements
    ax.text(x+w+10, y+h//2-20, f"Height: {height_pixels:.1f} px", 
            color='red', fontsize=10, backgroundcolor='white')
    ax.text(x+w+10, y+h//2+20, f"~{height_cm:.1f} cm", 
            color='red', fontsize=10, backgroundcolor='white')
    
    ax.set_title('Wave Height Measurement (Draw a box around the wave)')
    
    # Print the measurements
    print(f"Wave bounding box height: {height_pixels:.1f} pixels")
    print(f"Estimated wave height: {height_cm:.1f} cm (using scale: {pixel_to_cm_ratio:.4f} cm/pixel)")
    
    # Refresh the canvas
    fig.canvas.draw()

def adjust_scale(scale_cm):
    """Adjust the pixel to cm ratio based on user input."""
    global pixel_to_cm_ratio, wave_box
    
    if wave_box is not None:
        # Update the ratio based on the current box height and the provided scale
        height_pixels = wave_box[3]
        pixel_to_cm_ratio = scale_cm / height_pixels
        
        # Update the plot
        update_plot()
    else:
        print("Please select a wave box first before adjusting the scale.")

def main():
    global img, ax, fig, pixel_to_cm_ratio
    
    # Load the image
    image_path = 'practice wave.jpg'
    img = cv2.imread(image_path)
    
    if img is None:
        print(f"Error: Could not load image from {image_path}")
        return
    
    # Create a figure and connect the rectangle selector
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    ax.set_title('Draw a box around the wave to measure')
    
    # Create the RectangleSelector
    rs = RectangleSelector(
        ax, line_select_callback,
        drawtype='box', useblit=True,
        button=[1],  # Left mouse button only
        minspanx=5, minspany=5,
        spancoords='pixels',
        interactive=True)
    
    # Keep reference to the selector to prevent garbage collection
    plt.connect('key_press_event', rs)
    
    # Add instructions
    plt.figtext(0.5, 0.01, 
                "Instructions:\n"
                "1. Click and drag to draw a box around the wave\n"
                "2. Close this window when done\n"
                "3. Enter the estimated wave height in cm when prompted",
                ha='center', fontsize=10, bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    plt.show()
    
    # After the plot is closed, if a box was selected, ask for the scale
    if wave_box is not None:
        try:
            # Ask for the estimated wave height in cm
            scale_cm = float(input("Enter the estimated wave height in centimeters (e.g., 100): "))
            
            # Adjust the scale
            adjust_scale(scale_cm)
            
            # Create a new figure to show the final result
            fig, ax = plt.subplots(figsize=(10, 8))
            update_plot()
            
            # Save the result
            plt.savefig('wave_box_manual_result.png')
            plt.tight_layout()
            plt.show()
            
        except ValueError:
            print("Invalid input. Using default scale (100 cm wave height).")
            adjust_scale(100.0)  # Default to 100 cm if input is invalid
    else:
        print("No wave box was selected.")

if __name__ == "__main__":
    main() 