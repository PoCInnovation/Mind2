### Run the OpenBCI GUI
### Set Networking mode to LSL, FFT data type, and # Chan to 125
### Thanks to @Sentdex - Nov 2019
from pylsl import StreamInlet, resolve_stream
import numpy as np
import time
import matplotlib.pyplot as plt
from matplotlib import style
from collections import deque


last_print = time.time()
fps_counter = deque(maxlen=150)

# first resolve an EEG stream on the lab network
print("looking for an EEG stream...")
streams = resolve_stream('type', 'EEG')
# create a new inlet to read from the stream
inlet = StreamInlet(streams[0])

data = []
wait = 3 # wait 5 seconds before recording
total = 10
count = 0

# for i in range(25):  # how many iterations. Eventually this would be a while True
while True:

    channel_data = {}
    for i in range(8): # each of the 16 channels here
        sample, timestamp = inlet.pull_sample()
        if i not in channel_data:
            channel_data[i] = sample
        else:
            channel_data[i].append(sample)

    fps_counter.append(time.time() - last_print)
    last_print = time.time()
    cur_raw_hz = 1/(sum(fps_counter)/len(fps_counter))
    print(cur_raw_hz)
    tmp = []
    for i in range(8):
        tmp.append(np.array(channel_data[i][:60]))
    arr = np.array(tmp)
    if (np.shape(data)[0] == 25):
        if (count < wait):
            print(count, np.shape(data))
            data = []
        elif (count >= wait + total):
            break
        else:
            stamp = time.time() * 10
            stamp = int(stamp - (stamp % 1))
            name = "data" + str(stamp)
            np.save(name, data)
            print(name, np.shape(data))
            data = []
        count += 1
    data.append(arr)


# for chan in channel_data:
#     plt.plot(channel_data[chan][:60])
# plt.show()