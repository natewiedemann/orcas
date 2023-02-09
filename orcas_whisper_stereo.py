"""
for more info on Whisper, see: 
https://github.com/openai/whisper 

- import packages, load model ("large" model for accuracy), define functions

- specify root directory (Orchive). Filelist is created from all .wav files within
- files should be organized as root/year/tape/audio.wav
- Whisper ASR transcribes files (mono or stereo)
- raw transcript csv files saved into output folder structure (similar to synology)

- CSVs are read from output, text is processed and searched for Matrilines and Transients using RegEx functions

"""

import warnings
warnings.filterwarnings('ignore') # ignore message from Whisper about supported hardware
#warnings.filterwarnings(action='once')

import numpy as np
import pandas as pd
import soundfile as sf
import whisper
import jiwer
from jiwer import wer
import os
import time
import datetime
from fnmatch import fnmatch
import re
from pathlib import Path
import IPython.display as ipd


model = whisper.load_model("tiny", device="cpu") # tiny model for testing only
#model = whisper.load_model("large") # run on GPU, otherwise slow


# RegEx to find substrings matching naming conventions of Northern & Southern residents, e.g. a10, b201, etc.

def find_letter_number_substrings(string):
    pattern = re.compile(r'\b(\w)\s?(\d+)(s)?\s?')
    substrings = re.findall(pattern, string)
    output = sorted(list(set([substring[0] + substring[1] for substring in substrings])))
    return [elem for elem in output if elem[0].isalpha()]

# RegEx to find substrings matching naming conventions of Transients, e.g. T100, T203B2, etc.

def find_transient_substrings(string):
    pattern = r"T\s?\w+"
    substrings = re.findall(pattern, string)
    substrings = [x.replace(" ", "") for x in substrings]
    substrings = sorted(list(set(substrings)))
    substrings = [item for item in substrings if any(char.isdigit() for char in item)]
    return substrings

# Root folder containing tapes: root/year/tape/audio.wav

root = '/Users/natewiedemann/Desktop/orcas/audio' # change to local directory
pattern = "*.wav"

# create output savepath for transcripts

whattime = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
savepath = Path(os.getcwd() + '/orca_transcripts/' + whattime)
if not os.path.isdir(savepath):
    savepath.mkdir(parents=True)
if not os.path.isdir(f"{savepath}/temp"):
    os.mkdir(f"{savepath}/temp")

fileListOrcas = []
for path, subdirs, files in os.walk(root):
    for name in files:
        if fnmatch(name, pattern):
            fileListOrcas.append(os.path.join(path, name))

fileListOrcas = sorted(fileListOrcas)


# Read in files, transcribe with Whisper ASR

print(">>> Getting ASR Transcripts...\n")

t_start = time.time()

filename_List = []
transcript_raw_List = []

for i in range(0, len(fileListOrcas)):
    
    # create output path for transcript(s)

    output_dir = fileListOrcas[i].rsplit('/', 3)[-3:]
    output_path = Path(f"{savepath}/{output_dir[0]}/{output_dir[1]}/")
    if not os.path.isdir(output_path):
        output_path.mkdir(parents=True)

    # check if file is mono or stereo. If stereo, split to L and R channels and transcribe both.

    audiofile_data, samplerate = sf.read(fileListOrcas[i])

    stereo_flag = False
    nospeech_flag = False

    if len(audiofile_data.shape) > 1:
        audiofile_data_L = audiofile_data[:,0]
        audiofile_data_R = audiofile_data[:,1]
        sf.write(f"{savepath}/temp/audio_L.wav", audiofile_data_L, samplerate)
        sf.write(f"{savepath}/temp/audio_R.wav", audiofile_data_R, samplerate)
        stereo_flag = True

    if stereo_flag == True:
        n = 2
        file_input_path = [f"{savepath}/temp/audio_L.wav", f"{savepath}/temp/audio_R.wav"]
        filename_transcribed = [output_dir[2][0:-4] + "_L", output_dir[2][0:-4] + "_R"]
    else:
        n = 1
        file_input_path = [fileListOrcas[i]]
        filename_transcribed = [output_dir[2][0:-4] + "_mono"]
    
    
    for j in range(0,n):
        
        t0 = time.time()
        # transcribe with Whisper
        
        result_whisper = model.transcribe(file_input_path[j])
        result_whisper = result_whisper["text"]
        if len(result_whisper) == 0:
            result_whisper = "no speech detected" # should not be blank or empty string; otherwise error from .strip() when processing csv text later
            nospeech_flag = True

        # write to csv

        df=pd.DataFrame()
        df['filename'], df['whisper_transcript_raw'] = ([filename_transcribed[j]], [result_whisper])
        df.to_csv(f'{output_path}/{filename_transcribed[j]}.csv', sep='\t', index=False)
        
        # review results and audio
        
        print(filename_transcribed[j], "\n")
        #ipd.display(ipd.Audio(data=file_input_path[j], rate=samplerate, normalize = False)) # Jupyter Notebook audio player
        print(result_whisper)

        t1 = time.time()
        total_time = t1-t0
        print('\nProcessing time: {:.1f} seconds'.format(total_time))
        print('__________________________________________________________________ \n')

        # skip second stereo channel, if speech detected on first channel

        if j == 0 and nospeech_flag == False  and stereo_flag == True:
            t0 = time.time()
            result_whisper = "no speech detected (speech detected in opposite channel)"
            df=pd.DataFrame()
            df['filename'], df['whisper_transcript_raw'] = ([filename_transcribed[1]], [result_whisper])
            df.to_csv(f'{output_path}/{filename_transcribed[1]}.csv', sep='\t', index=False)
            print(filename_transcribed[1], "\n")
            print(result_whisper)
            t1 = time.time()
            total_time = t1-t0
            print('\nProcessing time: {:.1f} seconds'.format(total_time))
            print('__________________________________________________________________ \n')
            break

    
    
t_end = time.time()
total_time_all = (t_end-t_start)/60
print('\nASR Processing time (total): {:.1f} minutes'.format(total_time_all))


# Create list of transcript CSV files

root = savepath
pattern = "*.csv"

whattime = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
savepath_proc = f"{root}/processedTranscripts_{whattime}"

fileListOrcas_proc = []
for path, subdirs, files in os.walk(root):
    for name in files:
        if fnmatch(name, pattern):
            fileListOrcas_proc.append(os.path.join(path, name))

fileListOrcas_proc = sorted(fileListOrcas_proc)


# read in files from CSV

filename_List = []
transcript_raw_List = []
transcript_List = []
matriline_List = []
transient_List = []

for i in range(0, len(fileListOrcas_proc)):
        
    # read Whisper Raw transcript from CSV
    
    df_transcript = pd.read_csv(fileListOrcas_proc[i],sep='\t')
    filename = df_transcript["filename"].loc[0]
    result_whisper = df_transcript["whisper_transcript_raw"].loc[0]
    result_whisper_proc = (jiwer.RemovePunctuation()(result_whisper.strip()).lower())

    # find matrilines in transcript
    
    matriline = find_letter_number_substrings(result_whisper_proc)
    matriline = ', '.join(matriline) # convert to string for output format
    
    # find transient in transcript
    
    if result_whisper_proc.find("transient") != -1:
        transient = 'true'
    else:
        transient = 'false'
    
    filename_List.append(filename)
    transcript_raw_List.append(result_whisper)
    transcript_List.append(result_whisper_proc)
    matriline_List.append(matriline)
    transient_List.append(transient)

# write processed transcripts to new CSV (summary with matrilines)

df_proc = pd.DataFrame()
df_proc['filename'] = filename_List
df_proc['whisper_transcript_raw'] = transcript_raw_List
df_proc['whisper_transcript'] = transcript_List
df_proc['matrilines'] = matriline_List
df_proc['transients'] = transient_List

df_proc.to_csv(f'{savepath_proc}.csv', sep='\t', index=True)

print("done\n")
print("Output is saved at: ", savepath, "\n")
