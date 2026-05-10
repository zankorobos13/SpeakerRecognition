import tkinter as tk
from tkinter import filedialog
import torch
from torch.nn import Sequential, Linear, ReLU, CrossEntropyLoss, Conv1d, Dropout, BatchNorm1d
from torch.optim import SGD
from torch.utils.data import TensorDataset, DataLoader
import torchaudio
import soundfile as sf
import os

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

selected_file = None


# =========================================================
# ВАША ЛОГИКА МОДЕЛИ
# =========================================================
class StatsPooling(torch.nn.Module):
    def forward(self, x):
        # x: [B, C, T]
        mean = x.mean(dim=2)
        std = x.std(dim=2)
        return torch.cat([mean, std], dim=1)

class SpeakerModel(torch.nn.Module):
    def __init__(self):
        super().__init__()

        self.conv = Sequential(
            Conv1d(80, 128, kernel_size=5, padding=2),
            ReLU(),
            BatchNorm1d(128),

            Conv1d(128, 128, kernel_size=3, padding=1),
            ReLU(),
            BatchNorm1d(128),

            Conv1d(128, 128, kernel_size=3, padding=1),
            ReLU(),
            BatchNorm1d(128),
        )

        self.pool = StatsPooling()

        self.classifier = Sequential(
            Linear(256, 128),
            ReLU(),
            Dropout(0.3),

            Linear(128, 64),
            ReLU(),

            Linear(64, 10)
        )

    def forward(self, x):
        # x: [B, 80, T]
        x = self.conv(x)
        x = self.pool(x)
        x = self.classifier(x)
        return x

def get_data_from_wav(
    file_path,
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

    

    waveform, sample_rate = sf.read(file_path)

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

    

    return mel.squeeze(0)

def predict_voice(wav_path):
    """
    Здесь должна быть ваша логика.

    На вход:
        wav_path -> путь к wav файлу

    Должна вернуть:
        tensor shape [10]
        например:
        tensor([0.1, 0.05, ..., 0.2])

    """
    mel = get_data_from_wav(wav_path).to("cuda")
    mel = mel.unsqueeze(0)
    model = SpeakerModel().to("cuda")
    model.load_state_dict(torch.load("./tdnn_checkpoint.pt")["model"])
    model.eval()

    with torch.no_grad():
        y = model(mel)

    probs = torch.softmax(y, dim=1)
    return probs.squeeze(0)


# =========================================================
# ВЫВОД РЕЗУЛЬТАТОВ
# =========================================================

def show_prediction(probabilities):
    """
    probabilities: tensor [10]
    """

    # очистить старый вывод
    result_text.delete("1.0", tk.END)

    # лучший голос
    best_index = torch.argmax(probabilities).item()
    best_voice = VOICES[best_index]

    result_text.insert(tk.END, f"Predicted voice: {best_voice}\n\n")

    # вероятности
    for i in range(10):

        voice = VOICES[i]
        prob = probabilities[i].item()

        result_text.insert(
            tk.END,
            f"{voice}: {prob:.4f}\n"
        )


# =========================================================
# ВЫБОР ФАЙЛА
# =========================================================

def choose_file():

    global selected_file

    file_path = filedialog.askopenfilename(
        title="Select WAV file",
        filetypes=[("WAV files", "*.wav")]
    )

    if file_path:
        selected_file = file_path
        file_label.config(text=file_path)


# =========================================================
# ЗАПУСК МОДЕЛИ
# =========================================================

def run_model():

    if selected_file is None:
        result_text.delete("1.0", tk.END)
        result_text.insert(tk.END, "Select WAV file first")
        return

    # получить вероятности
    probs = predict_voice(selected_file)

    # показать результат
    show_prediction(probs)


# =========================================================
# GUI
# =========================================================

root = tk.Tk()
root.title("Speaker Recognition")
root.geometry("600x500")


title_label = tk.Label(
    root,
    text="Speaker Recognition",
    font=("Arial", 18)
)
title_label.pack(pady=10)


select_button = tk.Button(
    root,
    text="Choose WAV File",
    command=choose_file,
    width=20,
    height=2
)
select_button.pack(pady=10)


file_label = tk.Label(
    root,
    text="No file selected",
    wraplength=500
)
file_label.pack(pady=5)


predict_button = tk.Button(
    root,
    text="Predict Voice",
    command=run_model,
    width=20,
    height=2
)
predict_button.pack(pady=20)


result_text = tk.Text(
    root,
    height=20,
    width=60,
    font=("Consolas", 11)
)
result_text.pack(pady=10)


root.mainloop()