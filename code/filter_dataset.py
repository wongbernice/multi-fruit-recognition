import os
import shutil

# Get dataset paths
original_dataset_path = '../fruits-360'
original_training_path = '../fruits-360/Training'
original_testing_path = '../fruits-360/Testing'

filtered_dataset_path = '../filtered-dataset'
filtered_training_path = '../filtered-dataset/Training'
filtered_testing_path = '../filtered-dataset/Testing'

# ================================================ #
# Filter dataset to only include selected classes
# ================================================ #

# Get all classes in original dataset
all_training_classes = os.listdir(original_training_path)
all_testing_classes = os.listdir(original_testing_path)

# 10 selected fruit classes
classes = ['Apple', 'Banana', 'Blueberry', 'Cherry', 'Lemon', 'Mango', 'Orange', 'Pineapple', 'Strawberry', 'Watermelon']

# Filter Training Set
for fruit in classes:
    print(f"Filtering Training images: {fruit}")
    
    # Create new folder if it does not exist
    new_fruit_path = os.path.join(filtered_training_path, fruit)
    os.makedirs(new_fruit_path, exist_ok=True)

    # Loop through all original training folders
    for folder in os.listdir(original_training_path):

        # If folder matches selected fruit, copy to new folder location
        if fruit in folder:
            src_folder = os.path.join(original_training_path, folder) # original folder path

            # Copy all images from original folder to filtered folder
            for image in os.listdir(src_folder):
                src_image = os.path.join(src_folder, image)
                dst_image = os.path.join(new_fruit_path, image)
                shutil.copy2(src_image, dst_image)
    
print(f"Completed Training Filtering")


# Filter Testing Set
for fruit in classes:
    print(f"Filtering Testing images: {fruit}")

    # Create new folder if it does not exist
    new_fruit_path = os.path.join(filtered_testing_path, fruit)
    os.makedirs(new_fruit_path, exist_ok=True)

    # Loop through all original training folders
    for folder in os.listdir(original_testing_path):

        # If folder matches selected fruit, copy to new folder location
        if fruit in folder:
            src_folder = os.path.join(original_testing_path, folder) # original folder path

            # Copy all images from original folder to filtered folder
            for image in os.listdir(src_folder):
                src_image = os.path.join(src_folder, image)
                dst_image = os.path.join(new_fruit_path, image)
                shutil.copy2(src_image, dst_image)
    
    print(f"Completed Testing Filtering")
    

    