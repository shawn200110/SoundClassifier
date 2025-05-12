import torch
from cnn import CNNNetwork
from audio_dataset import DuckDataset
import torchaudio
from toolbox import preprocess_audio
from quack_trainer import AUDIO_DIR, NUM_SAMPLES, ANNOTATIONS_FILE, SAMPLE_RATE

# class_mapping = [
#     "rock",
#     "classical",
#     "jazz"
# ]

class_mapping = ["duck_quack", "not_a_duck_quack"]

def predict(model,input,target,class_mapping):
    model.eval()
    with torch.no_grad():
        predictions = model(input)
        # Tensor (1, 10) -> [ [0.1, .01, ..., 0.6]]
        predicted_index = predictions[0].argmax(0)
        predicted = class_mapping[predicted_index]
        expected = class_mapping[target]
    return predicted, expected 

### Run ##############################################################################################


if __name__ == '__main__':
    # load back the model
    model = CNNNetwork()
    state_dict = torch.load("duck_cnn.pth")
    model.load_state_dict(state_dict)

    # load duck dataset
    mel_spectrogram = torchaudio.transforms.MelSpectrogram(
        sample_rate=SAMPLE_RATE,
        n_fft=1024,
        hop_length=512,
        n_mels=64
        )

    dd = DuckDataset(ANNOTATIONS_FILE, AUDIO_DIR, mel_spectrogram, SAMPLE_RATE, NUM_SAMPLES,device="cpu")
    print("Accessed Dataset")

    # get a sample from the duck dataset for inference
    # idx = 45
    # input, target = dd[idx][0], dd[idx][1] #[batch_size,num_channels, freq, time]
    # input.unsqueeze_(0)
    # print("Predicting on:", dd.annotations.iloc[idx, 0])

    # file_path = "TestFiles/duck-attacked-sample.wav"
    # file_path = "TestFiles/male-laughter-mild-chuckles.wav"
    file_path = "TestFiles/bird-2.wav"
    
    input = preprocess_audio(file_path, mel_spectrogram, SAMPLE_RATE, NUM_SAMPLES)
    target = 1  # dummy label just for formatting (not used in final prediction)

    predicted, _ = predict(model, input, target, class_mapping)
    print(f"Prediction for '{file_path}': {predicted}")


    # make an inference
    predicted, expected = predict(model, input, target,class_mapping)

    print(f"Predicted: '{predicted}', Expected: '{expected}'")
