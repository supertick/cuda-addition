extern "C" __global__ void add(float* a, float* b, float* c, int n) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;

    printf("============== begin of kernel: c[%d] = %f\n", index, c[index]);

    if (index < n) {
        c[index] = a[index] + b[index];
    }

    printf("============== end of kernel: c[%d] = %f\n", index, c[index]);

}
