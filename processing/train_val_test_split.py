import os
import shutil
from sklearn.model_selection import train_test_split


def main():
    data_dir = '../data/processed/videoframe/'
    emotion_dir = os.listdir(os.path.join(data_dir, 'image'))
    seed = 42


    train_dir = os.path.join(data_dir, 'train')
    val_dir = os.path.join(data_dir, 'val')
    test_dir = os.path.join(data_dir, 'test')

    if not os.path.exists(train_dir):
        os.makedirs(train_dir, exist_ok=True)

    if not os.path.exists(val_dir):
        os.makedirs(val_dir, exist_ok=True)

    if not os.path.exists(test_dir):
        os.makedirs(test_dir, exist_ok=True)

    emotion_dir = ['Awe', 'Excitement', 'Fear', 'Sadness', 'Disgust', 'Anger', 'Amusement', 'Contentment']

    for emotion in emotion_dir:

        if not os.path.exists(os.path.join(train_dir, emotion)):
            os.makedirs(os.path.join(train_dir, emotion), exist_ok=True)

        if not os.path.exists(os.path.join(val_dir, emotion)):
            os.makedirs(os.path.join(val_dir, emotion), exist_ok=True)

        if not os.path.exists(os.path.join(test_dir, emotion)):
            os.makedirs(os.path.join(test_dir, emotion), exist_ok=True)

        video_list = os.listdir(os.path.join(data_dir, 'video', emotion))
        image_list = os.listdir(os.path.join(data_dir, 'image', emotion))
        n_image = len(image_list)

        if n_image < 1000:
            print(emotion, 'is excluded in the dataset due to the small frame numbers')
            continue

        elif n_image >= 1000:
            video_train, video_test = train_test_split(video_list, test_size=0.3, random_state=seed)
            video_val, video_test = train_test_split(video_test, test_size=2/3, random_state=seed)

            video_train = tuple([os.path.splitext(x)[0] for x in video_train])
            video_val = tuple([os.path.splitext(x)[0] for x in video_val])
            video_test = tuple([os.path.splitext(x)[0] for x in video_test])

            for image in image_list:
                if image.startswith(video_train):
                    original_path = os.path.join(data_dir, 'image', emotion, image)
                    destination_path = os.path.join(train_dir, emotion, image)
                    shutil.copy(original_path, destination_path)
                elif image.startswith(video_val):
                    original_path = os.path.join(data_dir, 'image', emotion, image)
                    destination_path = os.path.join(val_dir, emotion, image)
                    shutil.copy(original_path, destination_path)
                elif image.startswith(video_test):
                    original_path = os.path.join(data_dir, 'image', emotion, image)
                    destination_path = os.path.join(test_dir, emotion, image)
                    shutil.copy(original_path, destination_path)

if __name__ == "__main__":
    main()