# Dataset iterators (replace with actual data loading functions)
# import flickr30k
import os
from tensorflow.keras.preprocessing.image import load_img, img_to_array, save_img

# Datasets: 'name', 'load_data: returns iterator', 'prepare_data: some preprocessing'
# datasets = { #'flickr8k': (flickr8k.load_data, flickr8k.prepare_data),
            # 'coco': (coco.load_data, coco.prepare_data),
#            'flickr30k': (flickr30k.load_data, flickr30k.prepare_data),
#            }

# def get_dataset(name):
#    return datasets[name][0], datasets[name][1]

def preprocess_images(input_directory, output_directory):
        if not os.path.exists(output_directory):
            os.makedirs(output_directory)

        for img_name in os.listdir(input_directory):
            img_path = os.path.join(input_directory, img_name)
            img = load_img(img_path, target_size=(224, 224))
            img_array = img_to_array(img)

            preprocessed_img_path = os.path.join(output_directory, img_name)
            save_img(preprocessed_img_path, img_array)