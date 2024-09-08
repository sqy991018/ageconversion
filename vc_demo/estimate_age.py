import numpy as np
import torch
import torch.nn as nn
import torchaudio
from transformers import Wav2Vec2Processor
from transformers.models.wav2vec2.modeling_wav2vec2 import (
    Wav2Vec2Model,
    Wav2Vec2PreTrainedModel,
)

class ModelHead(nn.Module):
    r"""Classification head."""

    def __init__(self, config, num_labels):

        super().__init__()

        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.dropout = nn.Dropout(config.final_dropout)
        self.out_proj = nn.Linear(config.hidden_size, num_labels)

    def forward(self, features, **kwargs):

        x = features
        x = self.dropout(x)
        x = self.dense(x)
        x = torch.tanh(x)
        x = self.dropout(x)
        x = self.out_proj(x)

        return x


class AgeGenderModel(Wav2Vec2PreTrainedModel):
    r"""Speech emotion classifier."""

    def __init__(self, config):

        super().__init__(config)

        self.config = config
        self.wav2vec2 = Wav2Vec2Model(config)
        self.age = ModelHead(config, 1)
        self.gender = ModelHead(config, 3)
        self.init_weights()

    def forward(
            self,
            input_values,
    ):

        outputs = self.wav2vec2(input_values)
        hidden_states = outputs[0]
        hidden_states = torch.mean(hidden_states, dim=1)
        logits_age = self.age(hidden_states)
        logits_gender = torch.softmax(self.gender(hidden_states), dim=1)

        return hidden_states, logits_age, logits_gender


# 加载模型和处理器
device = 'cpu'
model_name = 'audeering/wav2vec2-large-robust-6-ft-age-gender'
processor = Wav2Vec2Processor.from_pretrained(model_name)
model = AgeGenderModel.from_pretrained(model_name)

# 从 test.wav 文件加载音频信号
# file_path = "./KIN/original_waveform.wav"
file_path = "./KIN/generated_waveform.wav"
audio_input, sample_rate = torchaudio.load(file_path)

# 如果音频有多个通道（例如立体声），我们只使用第一个通道
if audio_input.shape[0] > 1:
    audio_input = audio_input[0, :]

# 确保音频信号是以 float32 格式存储的 numpy 数组
signal = audio_input.numpy().astype(np.float32).reshape(1, -1)


def process_func(
    x: np.ndarray,
    sampling_rate: int,
    embeddings: bool = False,
) -> np.ndarray:
    r"""Predict age and gender or extract embeddings from raw audio signal."""

    # 通过处理器标准化信号
    y = processor(x, sampling_rate=sampling_rate)
    y = y['input_values'][0]
    y = y.reshape(1, -1)
    y = torch.from_numpy(y).to(device)

    # 通过模型进行预测
    with torch.no_grad():
        y = model(y)
        if embeddings:
            y = y[0]
        else:
            y = torch.hstack([y[1], y[2]])

    # 转换为 numpy 数组
    y = y.detach().cpu().numpy()
    result_str = np.array2string(y, formatter={'float_kind': lambda x: "%.8f" % x})

    return result_str


# 预测年龄和性别概率
print(' Age       child      male  female     ')
print(process_func(signal, 16000))

# Pooled hidden states of last transformer layer
# print(process_func(signal, sampling_rate, embeddings=True))
