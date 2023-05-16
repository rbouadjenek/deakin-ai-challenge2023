# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
#  Copyright (c) 2023. Reda Bouadjenek, Deakin University                      +
#     Email:  reda.bouadjenek@deakin.edu.au                                    +
#                                                                              +
#  Licensed under the Apache License, Version 2.0 (the "License");             +
#   you may not use this file except in compliance with the License.           +
#    You may obtain a copy of the License at:                                  +
#                                                                              +
#                 http://www.apache.org/licenses/LICENSE-2.0                   +
#                                                                              +
#    Unless required by applicable law or agreed to in writing, software       +
#    distributed under the License is distributed on an "AS IS" BASIS,         +
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.  +
#    See the License for the specific language governing permissions and       +
#    limitations under the License.                                            +
#                                                                              +
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++


import sys, os, h5py, json
import numpy as np
import tensorflow as tf
from tensorflow import keras

if __name__ == "__main__":
    if len(sys.argv) == 1:
        input_dir = '.'
        output_dir = '.'
    else:
        input_dir = os.path.abspath(sys.argv[1])
        output_dir = os.path.abspath(sys.argv[2])

    print("Using input_dir: " + input_dir)
    print("Using output_dir: " + output_dir)
    print(sys.version)
    print("Tensorflow version: " + tf.__version__)

    # Loading the model.
    model = 'model.h5'
    model_loaded = tf.keras.models.load_model('model.h5')
    print(model_loaded.summary())
    with h5py.File(model, mode='r') as f:
        vocab = f['vocab'][()]        
        try:
            answers = f['answers'][()]
        except:
            answers = ['yes', 'no']

    input_shape = model_loaded.input_shape
    image_size = np.array(input_shape[0][1:3])

    print('Size of inputs images: ' + str(image_size))

    # We now prepare the vocabulary that has been used.
    # Mapping tokens to integers
    token_to_num = keras.layers.StringLookup(vocabulary=vocab)
    # Mapping integers back to original tokens
    num_to_token = keras.layers.StringLookup(vocabulary=token_to_num.get_vocabulary(),
                                             invert=True)
    vocab_size = token_to_num.vocabulary_size()
    print(f"The size of the vocabulary = {token_to_num.vocabulary_size()}")
    print("Top 20 tokens in the vocabulary: ", token_to_num.get_vocabulary()[:20])

    # Read test dataset
    imgs_path_test = input_dir + '/simpsons_test/'
    q_val_file = imgs_path_test + 'questions.json'
    q_test = json.load(open(q_val_file))


    def preprocessing(questions, imgs_path):
        # Make sure the questions and annotations are alligned
        questions['questions'] = sorted(questions['questions'], key=lambda x: x['question_id'])
        q_out = []
        imgs_out = []
        q_ids = []
        # Preprocess questions
        for q in questions['questions']:
            # Preprocessing the question
            q_text = q['question'].lower()
            q_text = q_text.replace('?', ' ? ')
            q_text = q_text.replace('.', ' . ')
            q_text = q_text.replace(',', ' . ')
            q_text = q_text.replace('!', ' . ').strip()
            q_out.append(q_text)
            file_name = imgs_path + str(q['image_id']) + '.png'
            imgs_out.append(file_name)
            q_ids.append(q['question_id'])
        return imgs_out, q_out, q_ids


    imgs_test, q_test, q_ids_test = preprocessing(q_test, imgs_path_test)


    def encode_single_sample(img_file, q):
        ###########################################
        ##  Process the Image
        ##########################################
        # 1. Read image file
        img = tf.io.read_file(img_file)
        # 2. Decode the image
        img = tf.image.decode_jpeg(img, channels=3)
        # 3. Convert to float32 in [0, 1] range
        img = tf.image.convert_image_dtype(img, tf.float32)
        # 4. Resize to the desired size
        img = tf.image.resize(img, image_size)
        ###########################################
        ##  Process the question
        ##########################################
        # 5. Split into list of tokens
        word_splits = tf.strings.split(q, sep=" ")
        # 6. Map tokens to indices
        q = token_to_num(word_splits)
        # 7. Return an inputs to for the model
        return (img, q), 0


    # We define the batch size
    batch_size = 1
    # Define the test dataset
    test_dataset = tf.data.Dataset.from_tensor_slices((imgs_test, q_test))
    test_dataset = (test_dataset.map(encode_single_sample, num_parallel_calls=tf.data.AUTOTUNE)
                    .padded_batch(batch_size)
                    .prefetch(buffer_size=tf.data.AUTOTUNE)
                    )

    # Making predictions!
    y_proba = model_loaded.predict(test_dataset)
    y_predict = np.argmax(y_proba, axis=1)
    # Writting predictions to file.
    with open(os.path.join(output_dir, 'answers.txt'), 'w') as result_file:
        for i in range(len(y_predict)):
            result_file.write(str(q_ids_test[i]) + ',' + answers[y_predict[i]].decode('utf-8') + '\n')
