import os
import shutil

def filter_dataset(original_path, filtered_path, classes):
    '''
    Copies only selected classes from the original dataset to the filtered dataset
    
    original_path: path to original dataset
    filtered_path: path to filtered dataset
    classes: list of selected class (fruit) names
    '''

    for fruit in classes:
        print(f"Filtering images for {fruit}")
        
        # Create new class folder if it does not exist
        new_fruit_path = os.path.join(filtered_path, fruit)
        os.makedirs(new_fruit_path, exist_ok=True)

        # Loop through all original folders
        for folder in os.listdir(original_path):

            # If folder matches selected fruit, copy to new folder location
            if fruit in folder:
                src_folder = os.path.join(original_path, folder) # original folder path

                # Copy all images from original folder to filtered folder
                for image in os.listdir(src_folder):
                    src_image = os.path.join(src_folder, image)
                    dst_image = os.path.join(new_fruit_path, image)
                    shutil.copy2(src_image, dst_image)
        
        print(f"Completed filtering for {fruit}")

def main():
    # Get dataset paths
    original_training_path = '../fruits-360/Training'
    original_testing_path = '../fruits-360/Test'
    filtered_training_path = '../filtered-dataset/Training'
    filtered_testing_path = '../filtered-dataset/Testing'

    # 10 selected fruit classes
    classes = ['Apple', 'Banana', 'Blueberry', 'Cherry', 'Lemon', 'Orange', 'Peach', 'Pineapple', 'Strawberry', 'Watermelon']

    filter_dataset(original_training_path, filtered_training_path, classes)
    print(f"\nCompleted filtering for training")

    filter_dataset(original_testing_path, filtered_testing_path, classes)
    print(f"\nCompleted filtering for testing")

if __name__ == "__main__":
    main()
