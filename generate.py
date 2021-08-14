import os
import pickle
import numpy as np
import soundfile as sf
from soundgenerator import SoundGenerator
from autoencoder import VAE
from train import SPECTROGRAM_PATH

HOP_LENGTH = 256
SAVE_DIR_ORIGNAL = "sample/original"
SAVE_DIR_GENERATED = "sample/generated"
MIN_MAX_VALUES_PATH = "data/min_max_values.pkl"


def load_fsdd(spectrogram_path):
    x_train = []
    file_paths = []
    for root, _, file_names in os.walk(spectrogram_path):
        for file_name in file_names:
            file_path = os.path.join(root, file_name)
            spectrogram = np.load(file_path, allow_pickle=True)
            x_train.append(spectrogram)
            file_paths.append(file_path)
    x_train = np.array(x_train)
    x_train = x_train[..., np.newaxis]
    return x_train, file_paths


def select_spectrograms(spectrograms,
                        file_paths,
                        min__max_values,
                        num_spectrograms=2):
    #sampled_indexes = np.random.choice(range(len(spectrograms)), num_spectrograms)

    sampled_indexes = np.random.choice(range(len(spectrograms)), num_spectrograms)
    sampled_spectrogrmas = spectrograms[sampled_indexes]

    #sampled_spectrograms = spectrograms[sampled_indexes]
    file_paths = [file_paths[index] for index in sampled_indexes]
    sampled_min_max_values = [min__max_values[file_path] for file_path in file_paths]
    print(file_paths)
    print(sampled_min_max_values)
    return sampled_spectrograms, sampled_min_max_values


def save_signal(signals, save_dir, sample_rate=22050):
    for i, signal in enumerate(signals):
        save_path = os.path.join(save_dir, str(i) + ".wav")
        sf.write(save_path, signal, sample_rate)


if __name__ == "__main__":
    # initialise sound generator
    vae = VAE.load("model")
    sound_generator = SoundGenerator(vae, HOP_LENGTH)

    # load spectrogram + min max values
    with open(MIN_MAX_VALUES_PATH, "rb") as f:
        min_max_values = pickle.load(f)

    specs, file_paths = load_fsdd(SPECTROGRAM_PATH)

    # sample spectrogram + min max values
    sampled_specs, sampled_min_max_values = select_spectrograms(specs,
                                                                file_paths,
                                                                min_max_values,
                                                                5)

    # generate audio from sampled spectrogram
    signals, _ = sound_generator.generate(sampled_specs, sampled_min_max_values)

    # convert spectrogram samples to audio
    original_signals = sound_generator.convert_spectrograms_to_audio(
        sampled_specs, sampled_min_max_values
    )

    # save audio signals
    save_signal(signals, SAVE_DIR_GENERATED)
    original_signals(original_signals, SAVE_DIR_ORIGNAL)
