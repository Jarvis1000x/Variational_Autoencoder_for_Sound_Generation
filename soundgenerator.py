from preprocess import MinMaxNormaliser
import librosa

class SoundGenerator:
    # Sound generator is responsible for generating audio from spectrogram

    def __init__(self, vae, hop_length):
        self.vae = vae
        self.hop_length = hop_length
        self._min_max_normaliser = MinMaxNormaliser(0, 1)

    def generate(self, spectrograms, min_max_values):
        generated_spectrograms, latent_representations = self.vae.reconstruct(spectrograms)
        signals = self.convert_spectrograms_to_audio(generated_spectrograms, min_max_values)
        return signals, latent_representations

    def convert_spectrograms_to_audio(self, spectrograms, min_max_values):
        signals = []
        for spectrogram, min_max_value in zip(spectrograms, min_max_values):
            # 1 - reshape the log spectrogram
            log_spectrogram = spectrogram[:, :, 0]

            # 2 - apply denormalisation
            denorm_log_spec = self._min_max_normaliser.denormalise(
                log_spectrogram, min_max_value["min"], min_max_value["max"])

            # 3 - log spectrogram -> spectrogram
            spec = librosa.db_to_amplitude(denorm_log_spec)

            # 4 - apply griffin-lim algo
            signal = librosa.istft(spec, hop_length=self.hop_length)

            # 5 - append signal to signals
            signals.append(signal)

        return signals

