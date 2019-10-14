from keras.layers import BatchNormalization, Activation, Conv1D, MaxPooling1D, ZeroPadding1D, Input
from keras.models import Model
import numpy as np
import librosa

def preprocess(audio):
    audio *= 256.0  # SoundNet needs the range to be between -256 and 256
    # reshaping the audio data so it fits into the graph (batch_size, num_samples, num_filter_channels)
    audio = np.reshape(audio, (1, -1, 1))
    return audio


def load_audio(audio_file):
    sample_rate = 22050  # SoundNet works on mono audio files with a sample rate of 22050.
    audio, sr = librosa.load(audio_file, dtype='float32', sr=sample_rate, mono=True)
    audio = preprocess(audio)
    return audio



def remove_element(seq, element):
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
    model_weights = np.load('models/sound8.npy', encoding='latin1', allow_pickle=True).item()
    # model = Input(batch_input_shape=(1, None, 1))
    model = Input(shape=(1,1))

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
    last_layers = [
            {'name': 'conv8', 'num_filters': 1000, 'padding': 0,
                'kernel_size': 8, 'conv_strides': 2},
            {'name': 'conv8_2', 'num_filters': 401, 'padding': 0,
                'kernel_size': 8, 'conv_strides': 2},
            ]

    for x in filter_parameters:
        biases = model_weights[x['name']]['biases']
        weights_shape = remove_element(model_weights[x['name']]['weights'].shape, 1)
        weights = model_weights[x['name']]['weights'].reshape(weights_shape)
        model = ZeroPadding1D(padding=x['padding'])(model)
        model = Conv1D(x['num_filters'],
                         kernel_size=x['kernel_size'],
                         strides=x['conv_strides'],
                         padding='valid',
                         weights = [weights, biases]
                         )(model)

        gamma = model_weights[x['name']]['gamma']
        beta = model_weights[x['name']]['beta']
        mean = model_weights[x['name']]['mean']
        var = model_weights[x['name']]['var']


        model = BatchNormalization(weights=[gamma, beta, mean, var])(model)
        model = Activation('relu')(model)

        import ipdb; ipdb.set_trace()
        if 'pool_size' in x:
            model = MaxPooling1D(pool_size=x['pool_size'],
                                   strides=x['pool_strides'],
                                   padding='valid')(model)
    x = last_layers[0]
    weights = model_weights[x['name']]['weights'].reshape((8, 1024, 1000))
    biases = model_weights[x['name']]['biases']
    output_1 = Conv1D(x['num_filters'],
                         kernel_size=x['kernel_size'],
                         strides=x['conv_strides'],
                         weights=[biases, weights],
                         padding='valid')

    x = last_layers[1]
    output_2 = Conv1D(x['num_filters'],
                         kernel_size=x['kernel_size'],
                         strides=x['conv_strides'],
                         weights=[biases, weights],
                         padding='valid')

    return Model(inputs=model, outputs=[output_1, output_2])


def predict_scene_from_audio_file(audio_file):
    model = build_model()
    audio = load_audio(audio_file)
    return model.predict(audio)


def predictions_to_scenes(prediction):
    scenes = []
    # with open('categories/categories_places2.txt', 'r') as f:
    with open('categories/categories_imagenet.txt', 'r') as f:
        categories = f.read().split('\n')
        for p in range(prediction.shape[1]):
            scenes.append(categories[np.argmax(prediction[0, p, :])])
    return scenes


if __name__ == '__main__':
    #  SoundNet demonstration
    import sys
    filename = sys.argv[1]
    prediction = predict_scene_from_audio_file(filename)
    print(predictions_to_scenes(prediction))
