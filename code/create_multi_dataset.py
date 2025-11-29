import os
import random
import numpy as np
import cv2
 
def build_fruit_dict(dataset_path, classes):
    '''
    Store images paths for each fruit class in a dictionary
    
    dataset_path: path to training or testing dataset
    classes: list of selected class (fruit) names
    
    Returns: 
        fruit_dict: dictionary containing fruit image paths for each class
    '''
    # Create a dictionary to hold fruit image paths
    fruit_dict = {}

    # For each fruit class, store the images in a dictionary for easy access
    for fruit in classes:
        fruit_path = os.path.join(dataset_path, fruit) # get path to fruit folder
        fruit_images = os.listdir(fruit_path) # list all images in fruit folder
        fruit_dict[fruit] = [os.path.join(fruit_path, img) for img in fruit_images] # add all image paths to dictionary
    
    return fruit_dict

def create_multi_image(fruit_dict, classes):
    '''
    Create a multi-fruit image by randomly selecting and placing fruit images on a blank canvas

    fruit_dict: dictionary containing fruit image paths for each class
    classes: list of selected class (fruit) names 

    Returns:
        canvas: generated multi-fruit image
        label_vector: multi-hot label vector indicating presence of each fruit class
    '''
    # Create blank canvas for multi-fruit dataset images
    canvas_size = (280, 280, 3) # height, width, channels
    min_fruits = 2
    max_fruits = 6
    fruit_size = 80

    # Randomly select the number of fruits to place in an image (2-6)
    num_fruits = random.randint(min_fruits, max_fruits)

    # Randomly select fruit classes to place in an image
    selected_fruits = random.choices(classes, k=num_fruits)
    
    # Generate blank white canvas
    canvas = np.ones(canvas_size, dtype=np.uint8) * 255 

    # Predefined coordinates for each fruit index 0–5
    positions = [
        (10, 10),     # top-left
        (100, 10),    # top-middle
        (190, 10),    # top-right
        (10, 140),    # middle-left
        (100, 140),   # center
        (190, 140)    # middle-right
    ]

    used_images = {}  # To track used images and avoid duplicates

    # Iterate through selected fruits and place on canvas
    for i in range(len(selected_fruits)):

        # Initalize set if class is new
        if selected_fruits[i] not in used_images:
            used_images[selected_fruits[i]] = set()
        
        # Ensures no duplicate images for the same class
        while True:
            img_path = random.choice(fruit_dict[selected_fruits[i]]) # random image path from selected fruit class
            
            # Exit when unused image is found
            if img_path not in used_images[selected_fruits[i]]:
                used_images[selected_fruits[i]].add(img_path)
                break
        
        img = cv2.imread(img_path) # read image from path
        
        # Resize fruit images to 80 x 80
        img = cv2.resize(img, (fruit_size, fruit_size))

        # Paste fruit image onto canvas at (x, y)
        x_coord, y_coord = positions[i]
        canvas[y_coord : y_coord + fruit_size, x_coord : x_coord + fruit_size] = img
        
    # Create multi-hot label vector for selected fruits
    label_vector = []
    for fruit in classes:
        if fruit in selected_fruits:
            label_vector.append(1)
        else:
            label_vector.append(0)
    
    return canvas, label_vector

def create_dataset(dataset_path, fruit_dict, classes, total_images):
    '''
    Create multi-fruit dataset and corresponding label file
    
    dataset_path: path to save multi-fruit dataset
    fruit_dict: dictionary containing fruit image paths for each class
    classes: list of selected class (fruit) names
    total_images: total number of multi-fruit images to create
    '''
    # Create dataset paths if they do not exist
    os.makedirs(dataset_path, exist_ok=True)
    
    # Create txt file for labels
    labels_file = open(os.path.join(dataset_path, "labels_file.txt"), "w")

    for i in range(total_images):
        # Generate multi-fruit image and corresponding label vector
        canvas, label_vector = create_multi_image(fruit_dict, classes)

        # Save multi-fruit image
        image_filename = f"multi_fruit_{i+1}.jpg"
        image_path = os.path.join(dataset_path, image_filename)

        # Save image to path
        cv2.imwrite(image_path, canvas)

        # Save label vector to text file
        label_string = ''.join(map(str, label_vector))
        labels_file.write(f"{image_filename}\t{label_string}\n")
        
    labels_file.close()

def main():
    # Get paths for the filtered dataset
    filtered_training_path = '../filtered-dataset/Training'
    filtered_testing_path = '../filtered-dataset/Testing'

    # Get paths for the multi-fruit dataset
    multi_training_path = '../multi-fruit-dataset/Training'
    multi_testing_path = '../multi-fruit-dataset/Testing'

    # 10 selected fruit classes
    classes = ['Apple', 'Banana', 'Blueberry', 'Cherry', 'Lemon', 'Lychee', 'Orange', 'Peach', 'Strawberry', 'Watermelon']

    # Create dictionaries for multi-fruit training and testing datasets
    training_dict = build_fruit_dict(filtered_training_path, classes)
    testing_dict = build_fruit_dict(filtered_testing_path, classes)

    total_training_images = 500
    total_testing_images = 200

    # Create multi-fruit training and testing datasets
    create_dataset(multi_training_path, training_dict, classes, total_training_images)
    print(f"Completed Training Dataset")
    create_dataset(multi_testing_path, testing_dict, classes, total_testing_images)
    print(f"Completed Testing Dataset")

if __name__ == "__main__":
    main()