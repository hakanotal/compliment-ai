from infer_rvc_python import BaseLoader
from datasets import load_dataset
from transformers import pipeline
from pydub import AudioSegment
import noisereduce as nr
import scipy.io.wavfile
from tqdm import tqdm
import numpy as np
import traceback
import shutil
import torch
import re
import os

np.float = float   
output_dir = "audio_files" # tts audio files
final_dir = "final_nr" # final audio files after noise reduction

class Text2Audio:

    def __init__(self):
        
        # Load TTS model
        self.tts = pipeline("text-to-speech", model='microsoft/speecht5_tts', device=0)

        embeddings_dataset = load_dataset("Matthijs/cmu-arctic-xvectors", split="validation")

        self.speaker_embeddings = embeddings_dataset.filter(lambda example: example["filename"] == "cmu_us_bdl_arctic-wav-arctic_a0001")[0]['xvector']
        self.speaker_embeddings = torch.tensor(self.speaker_embeddings).unsqueeze(0)

        # Load RVC model
        model_path = './models/de_narrator.pth'
        index_path = './models/logs/de_narrator/added_IVF256_Flat_nprobe_1_de_narrator_v2.index'

        self.converter = BaseLoader(only_cpu=False, hubert_path=None, rmvpe_path=None)

        self.converter.apply_conf(
            tag="de_narrator",
            file_model=model_path,
            pitch_algo="rmvpe+",
            file_index=index_path,
            pitch_lvl=0,
            index_influence=0.66,
            respiration_median_filtering=3,
            envelope_ratio=0.25,
            consonant_breath_protection=0.33
        )


    def clear_directories(self):

        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        else:
            shutil.rmtree(output_dir)
            os.makedirs(output_dir)


        if not os.path.exists(final_dir):
            os.makedirs(final_dir)
        else:
            shutil.rmtree(final_dir)
            os.makedirs(final_dir)

    
    def generate_audio(self, input_text):
        for i, text in enumerate(tqdm(input_text.split('. '))):
            try:
                # Convert the chunk to audio using Hugging Face TTS
                if text[-1] != '.':
                    text += '.'

                audio_data = self.tts(text, forward_params={"speaker_embeddings": self.speaker_embeddings})

                # Extract the audio signal and sampling rate
                audio_signal = audio_data["audio"]
                sampling_rate = audio_data["sampling_rate"]

                # Save the audio file
                audio_file = f"{output_dir}/sentence_{i+1}.wav"
                scipy.io.wavfile.write(audio_file, sampling_rate, audio_signal)
                # print(f"Saved audio file: {audio_file}")

            except Exception as e:
                print(f"Error converting text to audio: {str(e)}")

        output_files = os.listdir(output_dir)
        output_files.sort(key=lambda x: int(re.sub(r'sentence_(\d+)\.wav', r'\1', x)))
        self.audio_files = [f"{output_dir}/{f}" for f in output_files]


    def apply_rvc(self):

        speakers_list = ["de_narrator"]

        result = self.converter(
            self.audio_files,
            speakers_list,
            overwrite=False,
            parallel_workers=2
        )

        return result
    
    def apply_noisereduce(self):

        result = []
        for audio_path in tqdm(self.audio_files):
            out_path = f"{final_dir}/{os.path.splitext(audio_path)[0].split('/')[-1]}.wav"

            try:
                # Load audio file
                audio = AudioSegment.from_file(f'{os.path.splitext(audio_path)[0]}_edited.wav')

                # Convert audio to numpy array
                samples = np.array(audio.get_array_of_samples())

                # Reduce noise
                reduced_noise = nr.reduce_noise(samples, sr=audio.frame_rate, prop_decrease=0.6)

                # Convert reduced noise signal back to audio
                reduced_audio = AudioSegment(
                    reduced_noise.tobytes(), 
                    frame_rate=audio.frame_rate, 
                    sample_width=audio.sample_width,
                    channels=audio.channels
                )

                # Save reduced audio to file
                reduced_audio.export(out_path, format="wav")
                result.append(out_path)

            except Exception as e:
                traceback.print_exc()
                print(f"Error noisereduce: {str(e)}")
                result.append(audio_path)

        return result
    

    def merge_audio_files(self):

        nr_audio_files = os.listdir(final_dir)
        nr_audio_files.sort(key=lambda x: int(re.sub(r'sentence_(\d+)\.wav', r'\1', x)))
        nr_audio_files = [f"{final_dir}/{f}" for f in nr_audio_files]

        combined_audio = AudioSegment.empty()

        for audio_file in tqdm(nr_audio_files):
            audio_segment = AudioSegment.from_file(audio_file)
            slowed_audio_segment = audio_segment._spawn(audio_segment.raw_data, overrides={"frame_rate": int(audio_segment.frame_rate * 0.9)})
            combined_audio += slowed_audio_segment
            combined_audio += AudioSegment.silent(duration=700)  # add silence

        combined_audio.export(f"final_output.wav", format="wav")

        self.clear_directories()
        
    
    def generate_audio_from_text(self, input_text):

        self.clear_directories()

        self.generate_audio(input_text)

        self.apply_rvc()

        self.apply_noisereduce()

        self.merge_audio_files()

        return "final_output.wav"
