from pydub import AudioSegment

def convert_to_wav(input_path, input_format, output_path=None):
    # Load the input file
    audio = AudioSegment.from_file(input_path, format=input_format)
    
    # Downsample: mono, 16kHz, 16-bit
    audio = audio.set_frame_rate(16000).set_channels(1).set_sample_width(2)

    if not output_path:
        output_path = input_path.rsplit('.', 1)[0] + '.wav'
    
    # Export to WAV
    audio.export(output_path, format="wav")
    print(f"File converted and saved as: {output_path}")
