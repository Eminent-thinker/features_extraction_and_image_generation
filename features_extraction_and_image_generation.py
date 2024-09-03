# importing all the required packages for classification of the images
# use the command "pip install -r requirements.txt" to install all the required packages

# import built-in package
import os
import json
import random
import shutil
import time

# Import scientific packages 
import tensorflow as tf
import numpy as np
import cv2

start_time = time.time()

# Avoiding OOM - (Out of Memory) Error in order to avoid inefficient memory management and to execute 
# multiple processes

gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)


def get_shape(vertices_number):
    if vertices_number >= 0:
        polygon = {
            0: 'circle', 1: 'Degenerate', 2: 'line', 3: 'Triangle', 4: 'Quadrilateral', 5: 'Pentagon',
            6: 'Hexagon', 7: 'Heptagon', 8: 'Octagon', 9: 'Nonagon', 10: 'Decagon', 11: 'Hendecagon',
            12: 'Dodecagon', 13: 'Tridecagon', 14: 'Tetradecagon', 15: 'Pentadecagon', 16: 'Hexadecagon',
            17: 'Heptadecagon', 18: 'Octadecagon', 19: 'Enneadecagon', 20: 'Icosagon', 21: 'Icosikaihenagon',
            22: 'Icosikaidigon', 23: 'Icosikaitrigon', 24: 'Icosikaitetragon', 25: 'Icosikaipegon'
        }
        return polygon[vertices_number]
    # else
    return None


def image_processing(image_path, save_path):
    # loading the image from the path, resize it and create different
    # format of the image in RGB
    input_image = cv2.imread(image_path)
#     resize_image = input_image
    resize_image = cv2.resize(input_image, (256, 256))
    
#     RGB_image = cv2.cvtColor(resize_image, cv2.COLOR_BGR2RGB)
    if resize_image is None:
        print(f"Error: Unable to load image from '{image_path}'")
        return None

    # split the input image into R, G, and B channels
#     red_image, green_image, blue_image = cv2.split(RGB_image)
    gray = cv2.cvtColor(resize_image, cv2.COLOR_RGB2GRAY)
    
#     image_and_labels = {"red image": red_image, "green image": green_image, "blue image": blue_image}

    # this stores all the features of each of the image format for red, green and blue
    # features ares : shape, centroid(COG), elevations,orientation angle
    features = list()

    # processing each format(RGB) of the image
    # count=0
    image_features = {}
#     for key, image in image_and_labels.items():

    # reducing the noise in the image by getting the x and y value of the features
    # detected
    smoothing = cv2.GaussianBlur(gray, (5, 5), 0)

    # detecting the edges in the image by variating the threshold values
    edges = cv2.Canny(smoothing, 30, 150, apertureSize = 3)

    # shape analysis and object detection and pattern recognition 
    contours, _ = cv2.findContours(edges.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    colours = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (0, 255, 255), (255, 0, 255), (192, 192, 192), 
                (128, 128, 128), (128, 0, 0), (128, 128, 0), (0, 128, 0), (128, 0, 128), (0, 128, 128), (0, 0, 128)]
    
    # Create a canvas to draw shapes on
    shape_canvas = np.zeros_like(resize_image) 
    for contour in contours:
        # checking for valid contours points before plotting or fitting an ellipse to the contours points
        if len(contour) >= 5:
            # calculate the area of the contour
            area = cv2.contourArea(contour)

            # approximate the contour to a simpler shape (polygon)
            length = cv2.arcLength(contour, True)
            epsilon = length * 0.02
            approx = cv2.approxPolyDP(curve=contour, epsilon=epsilon, closed=True)
            
            colour = random.choice(colours)
            
            # Draw the contour on the canvas
            cv2.drawContours(shape_canvas, [approx], 0, colour, 2)

            # determine the number of vertices, which help to identify differences shapes
            vertices_num = len(approx)

            # calculate the elevation(average intensity) of the pixels within the contour
            mask = np.zeros_like(gray)
            cv2.drawContours(mask, [contour], -1, 255, -1)
            mean_intensity = cv2.mean(resize_image, mask=mask)[0]

            # claculate the centeroids (center of gravity) of the shape
            M = cv2.moments(contour)
            if M['m00'] != 0:

                centroid_x = int(M['m10'] / M['m00'])
                centroid_y = int(M['m01'] / M['m00'])

            else:
                centroid_x = centroid_y = 0

            # calculate the orientation (angle) of the shape
            orientation_angle = cv2.fitEllipse(contour)[2]

            # get the shape based on the number of points(vertices)
            shape = get_shape(vertices_number=vertices_num)

            # store/save the features of each of image format processed for a particular 
            # image

            features.append({
                "shape": shape,
                "area": area,
                'length': length,
                "elevation": mean_intensity,
                "centroid": {"x": centroid_x, "y": centroid_y},
                "orientation": orientation_angle,
            })
            
    # Save the canvas as an image
    cv2.imwrite(save_path, shape_canvas)

    return features


def draw_polygon(image, n, length, centroid_x, centroid_y, orientation):
    side_length = length * 0.1
    
    colours = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (0, 255, 255), (255, 0, 255), (192, 192, 192), 
           (128, 128, 128), (128, 0, 0), (128, 128, 0), (0, 128, 0), (128, 0, 128), (0, 128, 128), (0, 0, 128)]
    
    colour = random.choice(colours)
    
    if n >= 3:
        points = []
        for i in range(n):
            x = centroid_x + side_length * np.cos((2 * np.pi * i/n) + np.deg2rad(orientation))
            y = centroid_y + side_length * np.sin((2 * np.pi * i/n) + np.deg2rad(orientation))
            points.append([int(x), int(y)])
        points = np.array(points, dtype = np.int32)
        cv2.polylines(image, [points], isClosed= True, color= colour, thickness= 2)
    elif n == 2:
        # handlle lines as a special case
        x1, y1 = centroid_x, centroid_y
        x2 = int(x1 + 2 * side_length * np.cos(np.deg2rad(orientation)))
        y2 = int(y1 + 2 * side_length * np.sin(np.deg2rad(orientation)))
        cv2.line(image, (x1,y1), (x2, y2), colour, 2)
    
    else:
        raise ValueError("Number of sides (n) must be 2 greater")
        
if __name__ == '__main__':
    # Specify the folder name in your workspace where the images are located
    image_dir = "data1"
    json_dir = "json_data"
    draw_dir = "draw_shape"
    contour_dir = "draw_contour"

    try:
        if os.path.exists(json_dir) and os.path.exists(draw_dir) and os.path.exists(contour_dir):
            # Remove the directory
            shutil.rmtree(json_dir)
            shutil.rmtree(draw_dir)
            shutil.rmtree(contour_dir)
        # Create the directory again
        os.makedirs(json_dir)
        os.makedirs(draw_dir)
        os.makedirs(contour_dir)
    except OSError as err:
        print(f"Error: {err}")

    # check if the folder exists
    if os.path.exists(image_dir) and os.path.isdir(image_dir) and os.path.exists(contour_dir):
        # get a list of files in the folder
        files = os.listdir(image_dir)

        # iterate through the files
        for i, file_name in enumerate(files):
            # check if the file is an image (you can add more file type checks if needed)
            if file_name.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp')):
                # get the full path of the image
                image_path = os.path.join(image_dir, file_name)
                
                contour_filename = os.path.splitext(file_name)[0] + '.png'
                save_path = os.path.join(contour_dir, contour_filename)
                
                start_time = time.time()

                # create a new dictionary for each image
                image_features = image_processing(image_path, save_path)
                
                json_filename = os.path.splitext(file_name)[0] + '.json'
                
                # create a unique JSON filename for each image using the image file name
                json_filename = os.path.join(json_dir, json_filename)

                # store the results in the JSON file
                with open(json_filename, 'w') as json_file:
                    json.dump(image_features, json_file, indent=4)

                end_time = time.time()
                print(f"Time taken to process '{file_name}': {end_time - start_time} seconds")

        print("Features have been saved to individual JSON files.")
    else:
        print(f"Error: The folder '{image_dir}' does not exist in your workspace.")

    # Step 2 of the project
    print("Wait:...... Draw the shapes of each features........")
    
    shapes = {'line' : 2, 'triangle' : 3, 'quadrilateral' : 4, 'pentagon': 5, 'hexagon': 6, 'heptagon': 7, 'octagon': 8, 'nonagon': 9, 'decagon': 10,
                   'hendecagon': 11, 'dodecagon': 12, 'tridecagon': 13, 'tetradecagon': 14, 'pentadecagon': 15,
                   'hexadecagon': 16, 'heptadecagon': 17, 'octadecagon': 18, 'enneadecagon': 19, 'icosagon': 20,
                   'icosikaihenagon': 21, 'icosikaidigon': 22, 'icosikaitrigon': 23, 'icosikaitetragon': 24,
                   'icosikaipegon': 25}
    
    # iterate through the image files in the image folder
    for filename in os.listdir(image_dir):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp')):
            # load the corresponding feature from the JSON file
            json_filename = os.path.splitext(filename)[0] + '.json'
            json_path = os.path.join(json_dir,  json_filename)
             
            # load and preprocess the corresponding image
            image_path = os.path.join(image_dir, filename)
            image = cv2.imread(image_path)
            canvas = np.zeros((400, 400, 3), dtype = np.uint8)
            
            with open(json_path, 'r') as json_file:
                feature_dict = json.load(json_file)
                
            for feature_data in feature_dict:

                # extract feature values from the feature dictionary
                shape_name = feature_data['shape']
                centroid_x = feature_data['centroid']['x']
                centroid_y = feature_data['centroid']['y']
                elevation = feature_data['elevation']
                area = feature_data['area']
                length = feature_data['length']
                orientation = feature_data['orientation']
                
                # determine the number of sides (n) based on the shape name
                n = shapes[shape_name.lower()]
                
                # draw the polygon based on the feature's properties
                draw_polygon(canvas, n, length, centroid_x + 50, centroid_y + 50, orientation)
                
            # save the image
            fname = os.path.splitext(filename)[0] + '.png'
            output_draw_image = os.path.join(draw_dir, fname)
            
            print(f"{output_draw_image} shape image draw")
            
            cv2.imwrite(output_draw_image, canvas)

