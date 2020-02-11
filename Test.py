from numba import cuda
import numpy as np
import time



@cuda.jit
def optimize(_in, out):
    start = cuda.grid(1)
    stride = cuda.gridsize(1)

    for i in range(start, _in.shape[0], stride):
        out[i][0] = (_in[i][4]) % 2
        out[i][1] = (_in[i][0] * _in[i][3]) % 2
        out[i][2] = (_in[i][1]) % 2
        out[i][3] = (_in[i][4] + 1) % 2
        out[i][4] = ((_in[i][2] + 1) * (_in[i][3] + 1)) % 2


x = np.array([[1, 0, 1, 0, 1], [1, 0, 1, 0, 1], [1, 0, 1, 0, 1], [1, 0, 1, 0, 1], [1, 0, 1, 0, 1]])

x_device = cuda.to_device(x)
out_device = cuda.device_array_like(x)

threads_per_block = 128
blocks_per_grid = 30

time1 = time.time()
optimize[blocks_per_grid, threads_per_block](x_device, out_device)
time2 = time.time()

print(time2-time1)
print(out_device.copy_to_host()[:6])



