import os
import numpy as np
import time
import paulPred
from pylsl import StreamInlet, resolve_stream

import socket
import struct
import traceback
import logging

def getFile(filepath):
    data = []
    file = np.load(filepath)
    for idx, line in enumerate(file):
        data.append(line)
    return data

def sendKeyStroke():
    ok = "oui"
    cmd = 'osascript -e \'tell application "System Events" to keystroke "p" ' "'"
    os.system(cmd)


def initLink ():
    s = socket.socket()
    socket.setdefaulttimeout(None)
    print('socket created ')
    port = 60000
    s.bind(('127.0.0.1', port)) #local host
    s.listen(30) #listening for connection for 30 sec?
    print('socket listensing ... ')
    while True:
        try:
            c, addr = s.accept()
            return c
        except Exception as e:
            logging.error(traceback.format_exc())
            print("error")
            c.sendall(bytearray([]))
            c.close()

def sending_and_reciveing(c, message):
        my_str_as_bytes = c.recv(4000) #received bytes
        # print(my_str_as_bytes.decode())

        bytes_to_send = str.encode(message)
        c.sendall(bytes_to_send) #sending back


# if __name__ == "__main__":
#     c = initLink()
#     model = paulPred.init("./acc100.00.pt")

#     t_end = time.time() + 15
#     while time.time() < t_end:

#         t_none = time.time() + 5
#         while time.time() < t_none:
#             time.sleep(0.05)
#             data = getFile("./data/none/none.npy")
#             thought = paulPred.real_time_prediction(model, data)
#             if thought == "go":
#                 sending_and_reciveing(c, "go")
#             else:
#                 sending_and_reciveing(c, "none")
#             print(thought)
#         t_go = time.time() + 4
#         while time.time() < t_go:
#             time.sleep(0.05)
#             data = getFile("./data/go/go.npy")
#             thought = paulPred.real_time_prediction(model, data)
#             if thought == "go":
#                 sending_and_reciveing(c, "go")
#             else:
#                 sending_and_reciveing(c, "none")
#             print(thought)
#     print("Done")
#     time.sleep(10)

def recieve_data(inlet):
    data = []
    while (np.shape(data)[0] < 25):
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
        data.append(arr)
    return data

if __name__ == "__main__":
    streams = resolve_stream('type', 'EEG')
    inlet = StreamInlet(streams[0])
    c = initLink()
    model = paulPred.init("./acc100.00.pt")

    t_end = time.time() + 60
    while time.time() < t_end:
        time.sleep(0.05)
        data = recieve_data(inlet)
        thought = paulPred.real_time_prediction(model, data)
        if thought == "go":
            sending_and_reciveing(c, "go")
        else:
            sending_and_reciveing(c, "none")
        print(thought)
    print("Done")
    time.sleep(10)
