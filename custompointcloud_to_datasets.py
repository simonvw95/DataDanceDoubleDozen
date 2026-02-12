import cv2
import numpy as np
import os

spec_dir = 'other_pointclouds/input_images/'

# define output directories for the PNG images and coordinates
output_dir = 'other_pointclouds/processed_pointclouds/'

# make sure the directories exist
os.makedirs(output_dir, exist_ok=True)

############################################################################################################
# very important variable, sets the number of points we want to use for each frame of the rickroll/datadance
N = 250
############################################################################################################

for img in os.listdir(spec_dir):

    # load the black-and-white image in grayscale
    image = cv2.imread(spec_dir + img, cv2.IMREAD_GRAYSCALE)

    # Define the range for light gray and black color (we want to keep these regions)
    black_threshold = 50  # Black pixel intensity threshold (0-255)
    light_gray_min = 100  # Minimum intensity for light gray
    light_gray_max = 220  # Maximum intensity for light gray

    # create a mask identifying the black and light gray areas
    mask = np.zeros_like(image, dtype=np.uint8)
    mask[(image <= black_threshold) | ((image >= light_gray_min) & (image <= light_gray_max))] = 255

    # get the coordinates of the black and light gray areas
    coordinates = np.column_stack(np.where(mask == 255))

    # evenly sample the coordinates from the black and light gray areas
    # we want to distribute N points over the identified area
    step_size = max(1, len(coordinates) // N)

    # select the coordinates for the dots (evenly spaced)
    selected_coords = coordinates[::step_size][:N]

    # create a blank white image to draw the dots
    output_image = np.ones_like(image, dtype=np.uint8) * 255  # White background

    # draw the dots (black color) on the image
    for (y, x) in selected_coords:
        cv2.circle(output_image, (x, y), radius=1, color=(0, 0, 0), thickness=-1)  # Black dot

    # save the image with dots as a .png file
    cv2.imwrite(output_dir + img.replace('.png', '_{}.png'.format(str(N))), output_image)

    # # compute centroid
    cx, cy = selected_coords.mean(axis = 0)
    # translate to origin
    X0 = selected_coords - np.array([cx, cy])
    # apply 90° CW rotation: (x',y') = (y, -x)
    R = np.array([[0, 1],
                  [-1, 0]])
    selected_coords = X0 @ R.T

    # normalize per axis (preserve aspect ratio)
    mins = selected_coords.min(axis=0)  # [min_x, min_y]
    maxs = selected_coords.max(axis=0)  # [max_x, max_y]

    # avoid divide by zero
    ranges = maxs - mins
    ranges[ranges == 0] = 1

    # scale to 0,100
    selected_coords = (selected_coords - mins) / ranges
    selected_coords = selected_coords * 100

    # manually scale to be closer to target x and y mean [54.26, 47.83]
    selected_coords *= 0.6
    selected_coords += 24

    print('X mean: ' + str(np.mean(selected_coords, axis = 0)[0]))
    print('Y mean: ' + str(np.mean(selected_coords, axis=0)[1]))
    print('X std: ' + str(np.std(selected_coords, axis=0)[0]))
    print('Y std: ' + str(np.std(selected_coords, axis=0)[1]))
    print('Correlation: ' + str(np.corrcoef(selected_coords[:, 0], selected_coords[:, 1])[0][1]))

    coords_file = os.path.join(output_dir, img.replace('.png', '_{}.csv'.format(str(N))))

    np.savetxt(coords_file, selected_coords, delimiter=',')

    print(f"Processed and saved: {img} and coordinates to {coords_file}")

    import matplotlib.pyplot as plt
    plt.scatter(x = selected_coords[:, 0], y = selected_coords[:, 1], s = 10)
    plt.xlim(0, 100)
    plt.ylim(0, 100)
    plt.gca().set_aspect('equal')
    # plt.set_aspect("equal", adjustable="box")  # perfect square axes
    plt.savefig(output_dir + 'matplotlib_' + img.replace('.png', '_{}.png'.format(str(N))))
    plt.close('all')
