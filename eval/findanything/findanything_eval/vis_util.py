import sys
import cv2
import numpy as np

def create_image_with_legend(colors, labels, image_size=(2000, 2000), spacing=15):
    """
    Creates an image with lines drawn in specified colors and labels next to them.

    Args:
        colors (numpy.ndarray): Array of colors with shape (n, 3), where each row is (R, G, B).
        labels (list of str): List of labels corresponding to each line.
        image_size (tuple): Size of the image (height, width).
        spacing (int): Spacing between each line and label.

    Returns:
        numpy.ndarray: The final image with lines and labels.
    """
    # Validate inputs
    assert len(colors) == len(labels), "Colors and labels must have the same length."

    # Create a blank image
    image = np.ones((image_size[0], image_size[1], 3), dtype=np.uint8) * 255

    # Define font properties
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.4
    thickness = 1
    margin = 10

    for i, (label, color) in enumerate(zip(labels, colors)):
        # Calculate y position for the line and label
        y_position = margin + i * spacing

        # Draw the line
        x_start = margin
        x_end = image_size[1] // 2
        cv2.line(image, (x_start, y_position), (x_end, y_position), color.tolist(), thickness=2)

        # Draw the label
        cv2.putText(image, label, (x_end + margin, y_position + 5), font, font_scale, (0, 0, 0), thickness)

    return image


def get_random_colors(num_colors):
    '''
    Generate random colors for visualization
    
    Args:
        num_colors (int): number of colors to generate
        
    Returns:
        colors (np.ndarray): (num_colors, 3) array of colors, in RGB, [0, 1]
    '''
    colors = []
    for i in range(num_colors):
        colors.append(np.random.rand(3))
    colors = np.array(colors)
    return colors