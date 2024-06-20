import os
import pandas as pd
import shutil
import cv2

# The videoframe dataset requires the processing of the data by splitting up the frames into images.

def save_all_frames(video_path, save_path, base_name, ext='jpg'):
    """
    Read the video files and return them as a frame

    :param video_path: Where the video files are saved.
    :param save_path: Where you would like to save the file.
    :param base_name: Original name of the video (the extension should me in 3 characters e.g. mp3)
    :param ext: extension of image file.
    :return: returns no output, but save all the image
    """
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        return

    os.makedirs(save_path, exist_ok=True)
    base_path = os.path.join(save_path, base_name)
    digit = len(str(int(cap.get(cv2.CAP_PROP_FRAME_COUNT))))

    n = 0
    while True:
        ret, frame = cap.read()
        if ret:
            cv2.imwrite('{}_{}.{}'.format(base_path, str(n).zfill(digit), ext), frame)
            n += 1
        else:
            return

def main():
    video_dir = '../data/raw/videoframe/'
    save_dir = '../data/processed/videoframe/'

    if not os.path.exists('../data/processed'):
        os.makedirs('../data/processed')

    if not os.path.exists(os.path.join(save_dir, 'video')):
        os.makedirs(os.path.join(save_dir, 'video'))

    if not os.path.exists(os.path.join(save_dir, 'image')):
        os.makedirs(os.path.join(save_dir, 'image'))

    emotion_list = ['Admiration', 'Adoration', 'Aesthetic Appreciation', 'Amusement', 'Anger', 'Anxiety',
                    'Awe', 'Awkwardness', 'Boredom', 'Calmness', 'Confusion', 'Craving',
                    'Disgust', 'Empathetic Pain', 'Entrancement', 'Excitement', 'Fear', 'Horror',
                    'Interest', 'Joy', 'Nostalgia', 'Relief', 'Romance', 'Sadness', 'Satisfaction',
                    'Sexual Desire', 'Surprise']

    df = pd.read_csv(os.path.join(video_dir, 'CowenKeltnerEmotionalVideos.csv'))

    video_list = df['Filename'].tolist()
    #emotion_class = df.iloc[:, 1:35].idxmax(axis=1).tolist()
    emotion_df = df[['Awe', 'Excitement', 'Fear', 'Sadness', 'Disgust', 'Anger', 'Amusement', 'contentment']]
    emotion_class = emotion_df.idxmax(axis=1).tolist()

    for idx, video in enumerate(video_list):
        original_path = os.path.join(video_dir, 'data', video)
        destination_path = os.path.join(save_dir, 'video', emotion_class[idx], video)

        if not os.path.exists(os.path.join(save_dir, emotion_class[idx])):
            os.makedirs(os.path.join(save_dir, 'video', emotion_class[idx]), exist_ok=True)
        shutil.copy(original_path, destination_path)

    for idx, video in enumerate(video_list):
        print('extracting a video:', video)
        os.makedirs(os.path.join(save_dir, 'image', emotion_class[idx]), exist_ok=True)
        save_all_frames(os.path.join(video_dir, 'data', video),
                        os.path.join(save_dir, 'image', emotion_class[idx]),
                        video[:-4],
                        'jpg')


if __name__ == "__main__":
    main()





