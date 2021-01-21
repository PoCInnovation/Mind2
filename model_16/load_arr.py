import numpy as np

arr = np.load("test16112132170.npy")
arr2 = np.load("test16112132260.npy")
for i in range(60):
    if (arr[0,5,i] != arr2[0,5,i]):
        print(arr[0,5,i], arr2[0,5,i])
    if (arr[23,5,i] != arr2[0,5,i]):
        print(arr[23,5,i], arr2[0,5,i])
for k in range(60):
    i = 16112132170
    end = 16112132260
    beg = 0
    while (i < end):
        prev = 0
        for j in range(25):
            arr3 = np.load("test" + str(i) + ".npy")
            cur = arr3[j, 2, k]
            if (prev != cur):
                print(cur)
                prev = cur
        i += 10
