import numpy as np

# num_l = np.zeros(4, np.int)
num_l = []
for i1 in range(8):
    for i2 in range(8):
        for i3 in range(8):
            for i4 in range(8):
                if sum([i1, i2, i3, i4]) == 8:
                    num_l.append([i1, i2, i3, i4])
                    print([i1, i2, i3, i4])
print(len(num_l))