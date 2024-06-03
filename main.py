import numpy as np
import matplotlib.pyplot as plt
from numpy.fft import fft, ifft, fftfreq

def analytical_signal(seg, N):
    X = fft(seg, N)
    h = np.zeros(N)
    h[0] = 1
    h[1:int(N/2)] = 2
    H = ifft(X * h)
    return H

def generation_AM_signal(fs, T, f_mod):
    t = np.arange(0, T, 1/fs)
    k = 1.5
    modulating_signal = np.sin(2 * np.pi * f_mod * t) # Модулирующий сигнал (низкочастотный стационарный сигнал)
#    np.random.seed(0) # Для воспроизводимости результатов
    carrier_noise = np.random.randn(len(t)) # Несущее колебание (шум)
    AM_noise_signal = (k + modulating_signal) * carrier_noise  # Принимаемый сигнал
    plt.figure(figsize=(10, 5))
    plt.plot(t, AM_noise_signal)
    plt.title(f'Принимаемый сигнал')
    plt.xlabel('Время [c]')
    plt.ylabel('Амплитуда')
    plt.show()
    return AM_noise_signal, t

def calc_envelope(AM_noise_signal, windowSize, t):
    numWindows = int(np.floor(len(AM_noise_signal) / windowSize))
    sum_envelope = np.array([])

    for i in range(numWindows):
        window = AM_noise_signal[(i)*windowSize : (i+1)* windowSize]
        H = abs(analytical_signal(window, windowSize))
        sum_envelope = np.hstack((sum_envelope, H))
    
    plt.figure(figsize=(10, 5))
    plt.plot(t, sum_envelope)
    plt.title('Огибающая')
    plt.xlabel('Время [c]')
    plt.ylabel('Амплитуда')
    plt.show()
    return sum_envelope

def calc_freq_envelope(envelope, fs, max_freq, N = False):
    if not N:
        N = int(np.log2(max_freq))

    new_fs = int(fs / N)
    envelope = envelope[::N]
    envelope_fft = fft(envelope)
    envelope_fft[0] = 0 
    envelope_spectrum = np.abs(envelope_fft)
    frequencies = fftfreq(len(envelope), 1/new_fs)
    dominant_frequency = frequencies[np.argmax(envelope_spectrum[:len(envelope)//2])]
    
    plt.figure(figsize=(10, 5))
    plt.plot(frequencies[:len(envelope)//new_fs * max_freq], envelope_spectrum[:len(envelope)//new_fs * max_freq])
    plt.title('Спектр огибающей')
    plt.xlabel('Частота (Гц)')
    plt.ylabel('Амплитуда')
    plt.grid(True)    
    plt.show()
    
    print(f"Доминирующая частота огибающей: {dominant_frequency} Гц")

############
def analysis_add_noise(fs, t, AM_noise_signal, windowSize, noise_level, max_freq, N = False):
    np.random.seed(0)
    noise_signal = np.random.randn(len(t)) * noise_level

    # Добавляем шум к чистому сигналу
    noisy_signal = AM_noise_signal + noise_signal

    plt.figure(figsize=(10, 5))
    plt.plot(t, noisy_signal)
    plt.title(f'Принимаемый сигнал с уровнем шума {noise_level}')
    plt.xlabel('Время [c]')
    plt.ylabel('Амплитуда')
    plt.show()

    windowSize = 512
    numWindows = int(np.floor(len(noisy_signal) / windowSize))

    sum_envelope = np.array([])

    for i in range(numWindows):
        window = noisy_signal[(i)*windowSize : (i+1)* windowSize]
        H = abs(analytical_signal(window, windowSize))
        sum_envelope = np.hstack((sum_envelope, H))

    plt.figure(figsize=(10, 5))
    plt.plot(t, sum_envelope)
    plt.title('Огибающая')
    plt.xlabel('Время [c]')
    plt.ylabel('Амплитуда')
    plt.show()

    if not N:
        N = int(np.log2(max_freq)) 

    new_fs = int(fs / N)
    sum_envelope = sum_envelope[::N]
    envelope_fft = fft(sum_envelope)
    envelope_fft[0] = 0 
    envelope_spectrum = np.abs(envelope_fft)
    frequencies = fftfreq(len(sum_envelope), 1/new_fs)
    dominant_frequency = frequencies[np.argmax(envelope_spectrum[:len(sum_envelope)//2])]
    
    plt.figure(figsize=(10, 5))
    plt.plot(frequencies[:len(sum_envelope)//new_fs * max_freq], envelope_spectrum[:len(sum_envelope)//new_fs * max_freq])
    plt.title('Спектр огибающей')
    plt.xlabel('Частота (Гц)')
    plt.ylabel('Амплитуда')
    plt.grid(True)
    plt.show()
    
    print(f"Доминирующая частота огибающей при уровне шума {noise_level}: {dominant_frequency} Гц")

############

def start():
    # Параметры сигнала
    fs = 2048  # Частота дискретизации
    T = 20  # Длительность сигнала в секундах
    f_mod = 5  # Частота модуляции, Гц (например, 5 Гц)
    windowSize = 512 #Размер анализируемого окна
    max_freq = 50 # Верхняя частота оценки частоты огибающей
    N = 8 # Параметр величины уменьшения частоты дискретизации (во сколько раз)
    AM_noise_signal, t = generation_AM_signal(fs, T, f_mod)
    envelope = calc_envelope(AM_noise_signal, windowSize, t)
    calc_freq_envelope(envelope, fs, max_freq, N)


#   Анализ при аддитивном шуме
    noise_level = [0.1, 0.5, 1.0, 2.0, 5.0]
    for level in noise_level:
        analysis_add_noise(fs, t, AM_noise_signal, windowSize, level, max_freq, N)


start()







