import numpy as np
import cmath
import matplotlib.pyplot as plt
import scikits.audiolab as audio



#function to calculate omega
def omega(x,y):
    return cmath.exp((2.0*cmath.pi*1j*y)/x)

#zero padding to make samples of the form 2^k
def padding_func(data):
   k = 0
   while k*512 < len(data):
      k += 1
   return np.concatenate((data, ([0] * (k*512 - len(data)))))

def fft(signal):
   n = len(signal)
   if n == 1:
      return signal
   else:

      Feven = fft([signal[i] for i in xrange(0, n, 2)])
      Fodd = fft([signal[i] for i in xrange(1, n, 2)])

      combined = [0] * n
      for m in xrange(n/2):
         combined[m] = Feven[m] + omega(n, -m) * Fodd[m]
         combined[m + n/2] = Feven[m] - omega(n, -m) * Fodd[m]

      return combined
#Hamming window

def ham_window(signal):
    new=[]

    for i in xrange(0,len(signal)):
        new.append(signal[i]*(0.54-(0.46*cmath.cos(float(2*cmath.pi*i/(len(signal)-1))))))

    return new

#----------------------------center frequeny function:------------------------

def center(signal):
    n=len(signal)

    for i in range(n):
        center+=(signal[i]*i)

    center/=n
    return center

#-----------------------------------------------------------------------------

#reading audio file
direc=['/home/kartik/Dataset/WAVs/new-classical/classical.00001.wav','/home/kartik/Dataset/WAVs/new-rock/rock.00001.wav','/home/kartik/Dataset/WAVs/new-jazz/jazz.00001.wav','/home/kartik/Dataset/WAVs/new-metal/metal.00001.wav']

for i in xrange(0,1):


    (inputSignal, samplingRate, bits) = audio.wavread(direc[i])


    new_signal=[]
    for p in xrange(0,len(inputSignal)-2,3):
        new_signal.append((inputSignal[p]+inputSignal[p+1]+inputSignal[p+2])/3)

    new_signal=np.array(new_signal)
    siglen=len(new_signal)
    frame_matrix=[]
    new=[]
    #--------------------------creating frames---------------------------------

    newest_signal=(padding_func(new_signal))

    signal_length=len(newest_signal)

    for i in xrange(0, (signal_length/512)):

        for j in xrange(0,512):

            new.append(newest_signal[(512*i)+j])

        frame_matrix.append(new)
        new=[]



    #fft implementation function

    #------------------------------Framing--------------------------------------



    frame_fft=[]
    frame_abs_output=[]
    roll_off=[]
    frame_PSD=[]

    for i in xrange(0, len(frame_matrix)):
        frame_fft.append(np.array(fft(frame_matrix[i])))

    for i in xrange(0, len(frame_matrix)):
        frame_abs_output.append(np.array([abs(frame_fft[i][j]) for j in xrange(0, 512)]))




    # ------------------------------------------Windowing---------------------------------------

    for i in xrange(0, len(frame_fft)):
        frame_fft[i]=ham_window(frame_fft[i])






    # ---------------------------------------- Feature Extraction--------------------
    #1. FFT
    fftoutput = np.array(fft(padding_func(new_signal)))

    abs_output=np.array([abs(fftoutput[i]) for i in xrange(0, len(padding_func(new_signal)))])



    #2. power spectral density of the signal:
    power_spectral_density=np.array([abs_output[i]**2 for i in xrange(0, len(abs_output))])

    for i in xrange(0, len(frame_fft)):
        frame_PSD.append(np.array([abs(frame_fft[i][j])**2 for j in xrange(0, 512)]))



    #3. Spectral Centroid

    spectral_centroid=[]
    centroid=0
    for i in xrange(0, len(frame_abs_output)):
        for j in range(512):
            centroid+=j*frame_abs_output[i][j]

        spectral_centroid.append(centroid)
        centroid=0



    #4. spectral roll_off


    Threshold_Energy=[]


    # find the spectral rolloff as the frequency position where the respective spectral energy is equal to c*totalEnergy
    mc=[]
    for i in range(len(frame_matrix)):
        Threshold_Energy.append(0.85*(np.sum(frame_abs_output[i] ** 2)))
        CumSum = np.cumsum(frame_abs_output[i]**2)

        [a, ]=np.nonzero(CumSum > Threshold_Energy[i])

        if len(a) > 0:
            mc.append(np.float64(a[0]) / (float(512)))
        else:
            mc.append(0.0)

    roll_off = np.mean(mc)


    #------------------Output Display-----------------------------


    # print ("No of bins: " + len(bin))



    print "\n"
    print ("Length of signal"+str(len(frame_matrix)))
    print ("Length of old signal"+str(len(frame_fft[5])))

    plt.plot(frame_fft[5])
    plt.ylabel('FFT(x)')
    plt.xlabel('frequency(n)')
    plt.show()


    plt.plot(mc)
    plt.ylabel('Roll-off values')
    plt.xlabel('frames(n)')
    plt.show()

    plt.plot(spectral_centroid)
    plt.ylabel('Spectral Centroid')
    plt.xlabel('frequency(n)')
    plt.show()

    plt.plot(frame_PSD[5])
    plt.ylabel('psd(x)')
    plt.xlabel('frequency(n)')
    plt.show()
