import os
import shutil

# The FI dataset requires no more processing. I make the directory name capitalized for the consistency.


def main():
    raw_dir = '../data/raw/FI/'
    process_dir = '../data/processed/'

    if not os.path.exists('../data/processed/FI/'):
        os.makedirs('../data/processed/FI/', exist_ok=True)

    dir_list = os.listdir(raw_dir)

    for dir in dir_list:
        original_path = os.path.join(raw_dir, dir)
        destination_path = os.path.join(process_dir, 'FI', dir.capitalize())

        shutil.copytree(original_path, destination_path)


if __name__ == "__main__":
    main()