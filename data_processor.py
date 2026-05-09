import os
import torch
import torchaudio
import soundfile as sf

VOICES = [
    "Zephyr",
    "Puck",
    "Charon",
    "Kore",
    "Fenrir",
    "Leda",
    "Orus",
    "Aoede",
    "Callirrhoe",
    "Autonoe"
]


def prepare_dataset(
    data_dir="./data",
    output_file="./data/dataset.pt",
    segment_seconds=3,
    n_mels=80
):
    """
    Подготовка датасета для speaker recognition.

    Что делает:
    1. Загружает wav файлы
    2. Режет на сегменты по 3 секунды
    3. Остатки выбрасывает
    4. Строит log-mel spectrogram
    5. Присваивает label 0-9
    6. Сохраняет dataset.pt

    Формат:
    {
        "X": tensor [N, n_mels, time],
        "y": tensor [N]
    }
    """

    all_features = []
    all_labels = []

    mel_transform = torchaudio.transforms.MelSpectrogram(
        n_mels=n_mels
    )

    for label, voice_name in enumerate(VOICES):

        voice_dir = os.path.join(data_dir, voice_name)

        if not os.path.exists(voice_dir):
            print(f"Skip missing folder: {voice_dir}")
            continue

        print(f"Processing {voice_name} -> label {label}")

        for filename in os.listdir(voice_dir):

            if not filename.endswith(".wav"):
                continue

            path = os.path.join(voice_dir, filename)

            waveform, sample_rate = sf.read(path)

            waveform = torch.tensor(waveform, dtype=torch.float32)

            # mono
            if waveform.ndim == 2:
                waveform = waveform.mean(dim=1)

            # [samples] -> [1, samples]
            waveform = waveform.unsqueeze(0)

            samples_per_segment = sample_rate * segment_seconds

            total_samples = waveform.shape[1]

            num_segments = total_samples // samples_per_segment

            for i in range(num_segments):

                start = i * samples_per_segment
                end = start + samples_per_segment

                segment = waveform[:, start:end]

                # mel spectrogram
                mel = mel_transform(segment)

                # log mel
                mel = torch.log(mel + 1e-6)

                # normalization
                mel = (mel - mel.mean()) / (mel.std() + 1e-9)

                # shape:
                # [n_mels, time]
                all_features.append(mel.squeeze(0))

                # label 0-9
                all_labels.append(label)

    X = torch.stack(all_features)
    y = torch.tensor(all_labels)

    dataset = {
        "X": X,
        "y": y,
        "voices": VOICES
    }

    torch.save(dataset, output_file)

    print()
    print("DONE")
    print(f"Saved: {output_file}")
    print(f"Samples: {len(X)}")
    print(f"X shape: {X.shape}")
    print(f"y shape: {y.shape}")

    return dataset

prepare_dataset()