import os
import json
import pandas as pd
import matplotlib.pyplot as plt


def convert_json_csv(dataset_path, target):
    df = pd.read_json(os.path.join(dataset_path, target)).rename(columns={0: "emotion", 1: "image", 2: "annotation"})
    annotation_df = pd.DataFrame(columns=['image_id', 'emotion', 'brightness', 'colorfulness', 'scene', 'object',
                                          'facial_expression', 'human_action'])
    annotation_file_list = df['annotation'].tolist()


    for file in annotation_file_list:
        annotation = pd.read_json(os.path.join(dataset_path, file), typ='series')
        annotation_df = pd.concat([annotation_df, annotation.to_frame().T])

    annotation_df.to_csv(os.path.join(dataset_path, target[:-5]+'_annotation.csv'))


def plot_distribution(dataset_path, annotation):
    path = os.path.join(dataset_path, annotation)
    df = pd.read_csv(path)
    df[['brightness', 'colorfulness']].plot.hist()
    plt.show()

def get_stats(dataset_path, annotation, attribute):
    path = os.path.join(dataset_path, annotation)
    df = pd.read_csv(path)
    count = df[attribute].value_counts()
    print('Here is the count summary for:',attribute+'\n', count)
    count.to_csv(os.path.join(dataset_path, annotation[:-4] +'_'+attribute+'_count.csv'))

def main():
    dataset_path = '../data/EmoSet-118K'
    annotation = 'train_annotation.csv'
    attribute ='scene'
    plot_distribution(dataset_path, annotation)
    get_stats(dataset_path, annotation, attribute)

if __name__ == "__main__":
    main()
