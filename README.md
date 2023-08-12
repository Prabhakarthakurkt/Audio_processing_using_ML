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


  Duration of whole audio is 33.52965986394558 seconds

# Visualising audio singal in the time domain

      plt.figure(figsize=(30,10)

      librose.display.waveplot(music, alpha=0.5)
      plt.ylim(-1,1))
      plt.title("simple music")


# Extracting Time Domain Audio Features

   The different types of time domain features are:

   ### Amplitude Envelope
   ### Zero Crossing Rate
   ### Root Mean Square Enerergy 


# Amplitude Evelope

  AE is the maximum value of all the samples in a frame

  Calculating an Amplitude Envelope

    FRAME_SIZE = 1024
    HOP_LENGTH = 512 ### used for overlapping frames

    def amplitud_envelope(signal, frame_size, hop_length):
        amplitude_envelope = []


        # calculate amplitude envelope for each frame
        for i in range(0, len(single),hop_length);
            amplitude_envelope_current_frame = max(single[i:i+frame_size])
            amplitude_envelope.append(amplitude_envelope_current_frame)

        reture np.array(amplitude_envelope)\

    AE_music = amplitude_envelope(music,FRAME_SIZE,HOP_LENGTH)

 if we want to see how many frames have been produced 

    print("The total number of frames produced :",len(AE_music))

 The total number of frame produced : 1445


# Visualising the AE

    frames = range(len(AE_music))
    t = librose.frames_to_time(frames, hop_length=HOP_LENGTH)
    plt.figure(figsize=(30,10))

    librosa.display.waveplot(music,alpha=5.0)
    plt.plot(t,AE_music,color="r")
    plt.ylim(-1,1))
    plt.title(sample music")


Text(0.5,1.0, 'simple music')

# Applications of Amplitude Envelope:

   ### gives max amplitude value in a frame 
   ### gives a rough idea of loudness
   ### used for genre classification or onset detection 

# Zero Crossing Rate:

  The is the rate at which a signal crossesm the negative value to zero and then to a possitive value

     ZCR_music = librose.feature.zero_crossing_rate(music, frame_length=FRAME_SIZE, hop_length=HOP_LENGTH)[0]


     frame = range(len(ZCR_music))
     t = librose.frames_to_time(frames_to_time(frames,hop_length =HOP_LENGTH)

     plt.figure(figsize=(30,10)
     plt.plot(t,ZCR_music,color='b')




# Applications of Zero Crossing Rate:

 ### can be used for recursive v/s pitch sound
 ### To find out voice/unvoice decisions 
 ### Can be used to find out monophonic pitch


# Root Mean Squre Energy:

  The is basically the root mean square of all the samples present in a frame 

  


























      
