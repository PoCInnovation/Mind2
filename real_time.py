### Run the OpenBCI GUI
### Set Networking mode to LSL, FFT data type, and # Chan to 125
### Thanks to @Sentdex - Nov 2019
from pylsl import StreamInlet, resolve_stream
import numpy as np
import mindPred
import sys

def connect_to_stream(model_path="./acc100.00.pt"):
    # first resolve an EEG stream on the lab network
    print(model_path)
    print("looking for an EEG stream...")
    streams = resolve_stream('type', 'EEG')
    # create a new inlet to read from the stream
    inlet = StreamInlet(streams[0])

    data = []
    wait = 3 # wait 3 seconds before recording
    total = 100
    count = 0
    model = mindPred.init(model_path)

    while True:

        channel_data = {}
        for i in range(8): # each of the 16 channels here
            sample, timestamp = inlet.pull_sample()
            if i not in channel_data:
                channel_data[i] = sample
            else:
                channel_data[i].append(sample)

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
                # stamp = time.time() * 10
                # stamp = int(stamp - (stamp % 1))
                # name = "data" + str(stamp)
                # np.save(name, data)
                # print(name, np.shape(data))
                mindPred.real_time_prediction(model, data)
                data = []
            count += 1
        data.append(arr)

if __name__ == "__main__":
    if (len(sys.argv) == 1):
        connect_to_stream()
    elif (len(sys.argv) == 2):
        connect_to_stream(sys.argv[1])