import os
import shutil
import pandas as pd

def main():
    raw_dir = '../data/raw/abstract_paintings/testImages_abstract/'
    process_dir = '../data/processed/'

    if not os.path.exists('../data/processed/abstract/'):
        os.makedirs('../data/processed/abstract/', exist_ok=True)

    df = pd.read_csv(os.path.join(raw_dir, 'ABSTRACT_groundTruth.csv'))
    img_list = df['Unnamed: 0'].tolist()
    emotion_class = df.iloc[:,1:].idxmax(axis=1).tolist()

    for idx, img in enumerate(img_list):
        img = img[1:-1]
        original_path = os.path.join(raw_dir, img)
        destination_path = os.path.join(process_dir, 'abstract', emotion_class[idx][1:-1], img)

        if not os.path.exists(os.path.join(process_dir, 'abstract', emotion_class[idx][1:-1])):
            os.makedirs(os.path.join(process_dir, 'abstract', emotion_class[idx][1:-1]), exist_ok=True)

        shutil.copy(original_path, destination_path)

if __name__ == "__main__":
    main()
