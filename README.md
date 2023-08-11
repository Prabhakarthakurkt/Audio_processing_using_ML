# Audio_processing_using_ML
We will be understanding and extracting different features from audio files which help us in performing ML operations on these audio file  

# Time Line of the project:

  -> Importing libraties and audio file 
  -> Understanding our audio file 
  -> Extracting Time Domain Audio Features 
  -> Fourier Transform and it's applications 
  -> Extracting Frequency Domain Audio Feature

## Installing Librose library

    ! pip install librosa


    import librosa
    import librosa.display
    import IPython.Display as ipd
    import numpy as np
    import matplotlib.pyplot as plt 

Loading our Audio File 

    ipd.Audio('put_audio_file_name_wav')

Reading our file with Librosa Library

     music , sr = librosa.load('put_audio_file_name_wav')

     print("show of the audio file:,music.shape)
     print("sample Rate of the audio file:",sr)


  shape of the audio file: (739329,)
  sample Rate of the audio file: 22055

  Duration of one sample in seconds


      sample_duration = 1 / sr
      print(f"Duration of one sample is{sample_duration:6f}seconds")

Duration of one sample in seconds 

      sample_duration = 1/ sr
      print("duration  of one sample is{sample_duration:6f}

Duration of one sample is 0.0000046

      # total number of sample in audio file 
      total_samples = len(music)
      tot_samples

739329


if we want to find out duration in seconds 

     duration = 1 / sr*tot_samples
     print(f"Duration of whole audio is {duration} seconds )


  Duration of whole audio 


























      
