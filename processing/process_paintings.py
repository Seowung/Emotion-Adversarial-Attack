import os
import shutil

def main():
    raw_dir = '../data/raw/abstract_paintings/testImages_artphoto/'
    process_dir = '../data/processed/'

    if not os.path.exists('../data/processed/artphoto/'):
        os.makedirs('../data/processed/artphoto/', exist_ok=True)

    img_list = os.listdir(raw_dir)

    for img in img_list:
        emotion = img.split('_')[0]

        original_path = os.path.join(raw_dir, img)
        destination_path = os.path.join(process_dir, 'artphoto', emotion.capitalize(), img)

        if not os.path.exists(os.path.join(process_dir, 'artphoto', emotion.capitalize())):
            os.makedirs(os.path.join(process_dir, 'artphoto', emotion.capitalize()))

        shutil.copy(original_path, destination_path)

if __name__ == "__main__":
    main()

