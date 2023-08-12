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

      RMS_music = librosa.feature.rms(music,frame_length=FRAME_SIZE,hop_;length=HOP_length)[0]

Visualising the RMS

    Frames = range(len(RMS_music))
    t = librose.frame_to_time(frames, hop_length=HOP_LENGTH)

    plt.figure(figsize=(30,10))

    librosa.display.waveplot(music,alpha=0.5)
    plt.plot(t,RMS_music,color="r")
    plt.ylim(-1,1)

    plt.title("simple music")

Text(0.5,1.0, 'simpele music')


# Application of RMS Energy:

   . give  RMS of all samples
   . This is also an indicator of loudness
   . can be used for audio segementation or music genre classification 

# Fourier Transformation:

  . Spectogram
  . Mel Frequencey Transform i.e plotting frequency spectrum

     fft_music = np.fft.fft(music)
     len(fft_music)
739329

     def plot_magnitude_spectrum(single,sr, title, f_ratio=1):
     fft_music = np.fft.fft(single)
     f_bins = int(len(fft_abcs)*f_ratio)

     plt.plot(f[:f_bins],fft_abs[:f_bins])
     plt.xlable('Frequence(HZ)')
     plt.title(title)


     plot_magnitude_spectrum(music, sr, "simple music",1)

# Plotting a spectogram

    #extracting short time fourier transform

    FRAME_SIZE = 2048
    HOT_SIZE = 512

    ssft = librosa.stft(music,n_fft=Frame_size,hop_length=HOP_SIZE)

    type(ssft[0][0])

numpy.complex64

we only want absolute values

    ssft_abs = np.abs(ssft)**2
    type(ssft_abs[0][0])

 numpy.float32

 # Visualizing 

     def plot_spectrogra(y,sr,hop_length, y_axis="linear"):
        plt.figure(figsize=(25,10)
        librose.display.specshow( y, 
                                  sr=sr,
                                  hop_length=hop_length,
                                  x_axis="time"
                                  y_axis=y_axis)
       plt.colorbar(format="%+2.f")

       plot_spectrogram(ssft_abs,sr,HOP_SIZE)


  This is because our audio file have low frequencis

       ### converting to log
       sfft_abs_log = librosa.power_to_db(sfft_abs)
       plot_spectrogram(sfft_abs_log,sr,HOP_SIZE,y_axis='log')


# Calculating Mel Frequency Cepstral Coefficients

   in sound processing, the mel-frequency cepstrum is a repesantation of the short-term power spectrum of a sound, based on a linear cosine transform of a log power spectrum on a nomlinear mel scale 
   frequency cepstral cofficient are coefficient that collectively make up an MFC.


     mfccs = librose.feature.mfcc(y=music,n_mfcc=13,sr=sr)

# Extracting 1st and 2nd order MFCCs

    delta_mfccs = librosa.feature.delta(mfccs)

    delta_mfccs = librosa.feature.delta(mfccs,order=2)

# Visulizing MFCCs

     plt.figure(figsize=(25,10))
     librosa.display.specshow(mfccs, 
                                 x_axis="time",
                                 sr=sr)
    plt.colorbar(format="%+2.f")
    plt.title('MFCCS')
    plt.show()


     plt.figure(figsize=(25,10))
     librosa.display.spaceshow(delta2_mfccs
                                x_axis="time",
                                sr=sr)
    plt.colorbar(format="%+2.f")
    plt.title('2nd order MFCCs')
    plt.show()



# Advamced or Frequency Domain Audio Features

   Different types of Frequency Domain Audio Feature are:

  . Band Energy Ratio
  . Spectral Centroid

# Band Energy Ratio:

  ratio of lower frequency bands to higher frequency bands

  it is used in comparison of energy in lower/higher frequency bands and it also measures how dominat low frequencies are 

  Application of Band Energy Ratio:

  . Music and Speech determination
  . Music classification 


# Spectral Centroid :

 The spectral centroid is a measure used in digital signal proccessing to characterise. it indicates where the center of mass of the spectrum is located. perceptually, it has a robust connection 
 with the impression of brightness of a sound.


  it is a measure of the brightness of the sound 

  calculates the weighted mean of frequencies

# Application of spectral Centroid:

  . Music Classification 
  . Audio Classification


  
 
  


























      
