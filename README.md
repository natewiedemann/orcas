# orcas

The environment consists of the following packages, if it's easier to install manually:
numpy
pandas
pip install -U openai-whisper
pip install jiwer
pip install soundfile

I had to use Python 3.8, as Whisper didn't work on my machine with 3.9.

ffmpeg (and possibly Rust) are requirements, see Whisper page for details:
https://github.com/openai/whisper

Using the code should be pretty straightforward:

- Modify Line 59 to your local directory with audio (e.g., root = '/Users/natewiedemann/Desktop/orcas_audio/' )
- The root directory should be organized as "orcas_audio/years/tapes/files.wav" (similar to structure on Synology)
- Line 35-36 selects the size of model to use. "Tiny" is good for quick tests but not very accurate.

Then just run the script and everything should flow from there, but let me know if anything is unclear or not working. 

It should transcribe all files, save the raw transcripts as CSVs, then read them back in and process the text. 
An output summary CSV includes all transcripts and any Matrilines found in the text search.

Stereo files: 
- Are split, the L channel is transcribed. 
- If speech is detected and transcribed on L channel, then R channel is skipped. The R transcript is replaced with "no speech detected (speech detected in opposite channel)"
- If no speech is detected on L channel, then R channel is transcribed. 
