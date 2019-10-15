import librosa
import numpy as np
import pandas as pd
from scipy.special import softmax
from keras.models import Model
from keras.layers import BatchNormalization, Activation, Conv1D, MaxPooling1D, ZeroPadding1D, Input


def preprocess(audio):
    audio *= 256.0  # SoundNet needs the range to be between -256 and 256
    # reshaping the audio data so it fits into the graph (batch_size, num_samples, num_filter_channels)
    audio = np.reshape(audio, (1, -1, 1))
    return audio


def load_audio(audio_file):
    # SoundNet works on mono audio files with a sample rate of 22050.
    sample_rate = 22050
    audio, sr = librosa.load(
        audio_file, dtype='float32', sr=sample_rate, mono=True)
    audio = preprocess(audio)
    return audio


def remove_element(seq, element):
    '''
    remove the first occurance of an element of
    a sequence (incl. immutable)

    orginally written by `zwer` on SO here:
    https://stackoverflow.com/questions/52617670/how-to-remove-the-first-instance-of-an-element-in-a-tuple

    pararms:
        seq (list-like): a sequence
        element (?): element to be removed from sequence
    '''
    try:
        index = seq.index(element)
        return seq[:index] + seq[index + 1:]
    except ValueError:  # element doesn't exist
        return seq


def build_model():
    """
    Builds up the SoundNet model and loads the weights from a given model file (8-layer model is kept at models/sound8.npy).
    :return:
    """
    model_weights = np.load('models/sound8.npy',
                            encoding='latin1', allow_pickle=True).item()
    input_layer = Input(shape=(None, 1), name='input')

    filter_parameters = [{'name': 'conv1', 'num_filters': 16, 'padding': 32,
                          'kernel_size': 64, 'conv_strides': 2,
                          'pool_size': 8, 'pool_strides': 8},

                         {'name': 'conv2', 'num_filters': 32, 'padding': 16,
                          'kernel_size': 32, 'conv_strides': 2,
                          'pool_size': 8, 'pool_strides': 8},

                         {'name': 'conv3', 'num_filters': 64, 'padding': 8,
                          'kernel_size': 16, 'conv_strides': 2},

                         {'name': 'conv4', 'num_filters': 128, 'padding': 4,
                          'kernel_size': 8, 'conv_strides': 2},

                         {'name': 'conv5', 'num_filters': 256, 'padding': 2,
                          'kernel_size': 4, 'conv_strides': 2,
                          'pool_size': 4, 'pool_strides': 4},

                         {'name': 'conv6', 'num_filters': 512, 'padding': 2,
                          'kernel_size': 4, 'conv_strides': 2},

                         {'name': 'conv7', 'num_filters': 1024, 'padding': 2,
                          'kernel_size': 4, 'conv_strides': 2},
                         ]

    model = ZeroPadding1D(padding=32)(input_layer)
    for x in filter_parameters:
        biases = model_weights[x['name']]['biases']
        weights_shape = remove_element(
            model_weights[x['name']]['weights'].shape, 1)
        weights = model_weights[x['name']]['weights'].reshape(weights_shape)
        if 'conv1' not in x['name']:
            model = ZeroPadding1D(padding=x['padding'])(model)
        model = Conv1D(x['num_filters'],
                       kernel_size=x['kernel_size'],
                       strides=x['conv_strides'],
                       padding='valid',
                       weights=[weights, biases]
                       )(model)

        gamma = model_weights[x['name']]['gamma']
        beta = model_weights[x['name']]['beta']
        mean = model_weights[x['name']]['mean']
        var = model_weights[x['name']]['var']

        model = BatchNormalization(weights=[gamma, beta, mean, var])(model)
        model = Activation('relu')(model)

        if 'pool_size' in x:
            model = MaxPooling1D(pool_size=x['pool_size'],
                                 strides=x['pool_strides'],
                                 padding='valid')(model)

    conv8 = {'name': 'conv8', 'num_filters': 1000, 'padding': 0,
                      'kernel_size': 8, 'conv_strides': 2}
    conv8_2 = {'name': 'conv8_2', 'num_filters': 401, 'padding': 0,
                      'kernel_size': 8, 'conv_strides': 2}

    weights = model_weights[conv8['name']]['weights'].reshape((8, 1024, 1000))
    biases = model_weights[conv8['name']]['biases']
    output_1 = Conv1D(conv8['num_filters'],
                      kernel_size=conv8['kernel_size'],
                      strides=conv8['conv_strides'],
                      weights=[weights, biases],
                      padding='valid')(model)

    weights = model_weights[conv8_2['name']
                            ]['weights'].reshape((8, 1024, 401))
    biases = model_weights[conv8_2['name']]['biases']
    output_2 = Conv1D(conv8_2['num_filters'],
                      kernel_size=conv8_2['kernel_size'],
                      strides=conv8_2['conv_strides'],
                      weights=[weights, biases],
                      padding='valid')(model)

    return Model(inputs=input_layer, outputs=[output_1, output_2])


def predict_scene_from_audio_file(audio_file):
    model = build_model()
    audio = load_audio(audio_file)
    return model.predict(audio)


def predictions_to_scenes(prediction):
    with open('categories/categories_places2.txt', 'r') as f:
        places = np.array(f.read().strip().split('\n'))
    with open('categories/categories_imagenet.txt', 'r') as f:
        imagenet = np.array(f.read().strip().split('\n'))

    object_distro = softmax(prediction[0].reshape(-1, 1000), axis=1)
    place_distro = softmax(prediction[1].reshape(-1, 401), axis=1)

    return pd.DataFrame(object_distro, columns=imagenet), pd.DataFrame(place_distro, columns=places)


if __name__ == '__main__':
    #  SoundNet demonstration
    import sys
    filename = sys.argv[1]
    prediction = predict_scene_from_audio_file(filename)
    print(predictions_to_scenes(prediction))
