import os, shutil, argparse, json, logging
import pandas as pd


def process_emoset(dataset_dir: str) -> None:
    '''
    Process the EmoSet-118K dataset into folders based on the provided
    train, val, and test splits and further based on emotion class.
    
    Args:
    dataset_dir (str): directory of corresponding datasets in `/red/`
    '''
    
    raw_dir = os.path.join(dataset_dir, 'raw/EmoSet-118K')
    processed_dir = os.path.join(dataset_dir, 'processed/EmoSet-118K')
    
    with open(os.path.join(raw_dir, 'train.json'), 'r') as f:
        train_split = json.load(f)
        
    with open(os.path.join(raw_dir, 'val.json'), 'r') as f:
        val_split = json.load(f)
        
    with open(os.path.join(raw_dir, 'test.json'), 'r') as f:
        test_split = json.load(f)
                               
    if not os.path.exists(os.path.join(processed_dir, 'train')):
        os.makedirs(os.path.join(processed_dir, 'train'))
    
    if not os.path.exists(os.path.join(processed_dir, 'val')):
        os.makedirs(os.path.join(processed_dir, 'val'))
                 
    if not os.path.exists(os.path.join(processed_dir, 'test')):
        os.makedirs(os.path.join(processed_dir, 'test'))
    
    for datapoint in train_split:
        emotion_class_dir = os.path.join(processed_dir, 'train', datapoint[0].capitalize())
        train_img_dir = os.path.join(raw_dir, datapoint[1])
                               
        if not os.path.exists(emotion_class_dir):
            os.makedirs(emotion_class_dir)
                               
        shutil.copy(train_img_dir, emotion_class_dir)
                 
    for datapoint in val_split:
        emotion_class_dir = os.path.join(processed_dir, 'val', datapoint[0].capitalize())
        val_img_dir = os.path.join(raw_dir, datapoint[1])
                               
        if not os.path.exists(emotion_class_dir):
            os.mkdir(emotion_class_dir)
                               
        shutil.copy(val_img_dir, emotion_class_dir)
                 
    for datapoint in test_split:
        emotion_class_dir = os.path.join(processed_dir, 'test', datapoint[0].capitalize())
        test_img_dir = os.path.join(raw_dir, datapoint[1])
                               
        if not os.path.exists(emotion_class_dir):
            os.mkdir(emotion_class_dir)
                               
        shutil.copy(test_img_dir, emotion_class_dir)

        
def process_abstract(dataset_dir: str) -> None:
    raw_dir = os.path.join(dataset_dir, 'raw/abstract_paintings/testImages_abstract')
    processed_dir = os.path.join(dataset_dir, 'processed/abstract')

    df = pd.read_csv(os.path.join(raw_dir, 'ABSTRACT_groundTruth.csv'))
    img_list = df.iloc[:, 0].tolist()
    emotion_class = df.iloc[:,1:].idxmax(axis=1).tolist()

    for idx, img in enumerate(img_list):
        img = img[1:-1]
        original_path = os.path.join(raw_dir, img)
        destination_path = os.path.join(processed_dir, emotion_class[idx][1:-1], img)

        if not os.path.exists(os.path.join(processed_dir, emotion_class[idx][1:-1])):
            os.makedirs(os.path.join(processed_dir, emotion_class[idx][1:-1]), exist_ok=True)

        shutil.copy(original_path, destination_path)


def process_artphoto(dataset_dir: str) -> None:
    raw_dir = os.path.join(dataset_dir, 'raw/abstract_paintings/testImages_artphoto')
    processed_dir = os.path.join(dataset_dir, 'processed/artphoto')

    if not os.path.exists(processed_dir):
        os.mkdir(processed_dir)

    img_list = [f for f in os.listdir(raw_dir) if f.endswith('.jpg')]

    for img in img_list:
        emotion = img.split('_')[0]

        original_path = os.path.join(raw_dir, img)
        destination_path = os.path.join(processed_dir, emotion.capitalize(), img)

        if not os.path.exists(os.path.join(processed_dir, emotion.capitalize())):
            os.makedirs(os.path.join(processed_dir, emotion.capitalize()))

        shutil.copy(original_path, destination_path)


def process_caer(dataset_dir: str) -> None:
    raw_dir = os.path.join(dataset_dir, 'raw/CAER-S')
    processed_dir = os.path.join(dataset_dir, 'processed/CAER-S')
    
    dir_list = os.listdir(os.path.join(raw_dir, 'train'))
                            
    for dir_ in dir_list:
        original_train_path = os.path.join(raw_dir, 'train', dir_)
        original_test_path = os.path.join(raw_dir, 'test', dir_)
        destination_path = os.path.join(processed_dir, dir_)
        
        if not os.path.exists(destination_path):
            os.makedirs(destination_path)

        for file in os.listdir(original_train_path):
            s = os.path.join(original_train_path, file)
            d = os.path.join(destination_path)
            shutil.copy(s, d)

        for file in os.listdir(original_test_path):
            s = os.path.join(original_train_path, file)
            d = os.path.join(destination_path)
            shutil.copy(s, d)
        
        
def process_dvisa(dataset_dir: str) -> None:
    raw_dir = os.path.join(dataset_dir, 'raw/D-ViSA')
    processed_dir = os.path.join(dataset_dir, 'processed/D-ViSA')
        
    df = pd.read_csv(os.path.join(raw_dir, 'D-ViSA.csv'))
    img_list = df.iloc[:, 0].tolist()
    emotion_class = df.iloc[:, 1].tolist()

    for idx, img in enumerate(img_list):
        original_path = os.path.join(raw_dir, 'data', img)
        destination_path = os.path.join(processed_dir, emotion_class[idx].capitalize(), img)

        if not os.path.exists(os.path.join(processed_dir, emotion_class[idx].capitalize())):
            os.makedirs(os.path.join(processed_dir, emotion_class[idx].capitalize()), exist_ok=True)

        shutil.copy(original_path, destination_path)


def process_fi(dataset_dir: str) -> None:
    raw_dir = os.path.join(dataset_dir, 'raw/FI')
    processed_dir = os.path.join(dataset_dir, 'processed/FI')

    if not os.path.exists(processed_dir):
        os.mkdir(processed_dir)

    dir_list = os.listdir(raw_dir)

    for dir_ in dir_list:
        original_path = os.path.join(raw_dir, dir_)
        destination_path = os.path.join(processed_dir, dir_.capitalize())
                        
        shutil.copytree(original_path, destination_path)


def main():
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--dataset', type=str, help='Which dataset to process')
    args = parser.parse_args()
    
    data_dir = '/red/ruogu.fang/share/emotion_adversarial_attack/data'
    
    match args.dataset:
        case 'emoset':
            process_emoset(data_dir)
        case 'abstract':
            process_abstract(data_dir)
        case 'artphoto':
            process_artphoto(data_dir)
        case 'CAER-S':
            process_caer(data_dir)
        case 'D-ViSA':
            process_dvisa(data_dir)
        case 'FI':
            process_fi(data_dir)
        case _:
            print('Other datasets currently available.')


if __name__ == '__main__':
    main()