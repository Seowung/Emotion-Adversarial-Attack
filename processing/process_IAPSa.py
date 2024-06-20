import os
import shutil
import pandas as pd



def main():
    raw_dir = '../data/raw/IAPS_all/'
    process_dir = '../data/processed/IAPSa'

    if not os.path.exists(process_dir):
        os.makedirs(process_dir,exist_ok=True)

    df_neg = pd.read_csv('../data/raw/IAPSa/Mikels2005negativenorms.csv')
    df_pos = pd.read_csv('../data/raw/IAPSa/Mikels2005positivenorms.csv')

    df_pos = df_pos[['Picture', 'AmusMean', 'AweMean', 'ContMean', 'ExciMean']]
    df_neg = df_neg[['Picture', 'AngrMean', 'DisgMean', 'FearMean', 'SadnMean']]

    df = pd.concat([df_neg, df_pos])
    df = df.rename(columns={'AmusMean': 'Amusement', 'AweMean': 'Awe', 'ContMean': "Contentment", "ExciMean": "Excitement",
                            'AngrMean': 'Anger', 'DisgMean': 'Disgust', 'FearMean': 'Fear', 'SadnMean': 'Sadness'})


    img_list = df['Picture'].tolist()
    emotion = df.iloc[:, 1:].idxmax(axis=1).tolist()

    for idx, img in enumerate(img_list):
        if img == 2819: # this is not included in the file.
            continue
        elif str(img).endswith('0'):
            img = str("{:.0f}".format(img)) + '.jpg'
        else:
            img = str("{:.1f}".format(img)) + '.jpg'

        original_path = os.path.join(raw_dir, img)
        destination_path = os.path.join(process_dir, emotion[idx], img)

        if not os.path.exists(os.path.join(process_dir, emotion[idx])):
            os.makedirs(os.path.join(process_dir, emotion[idx]), exist_ok=True)

        shutil.copy(original_path, destination_path)

if __name__ == "__main__":
    main()