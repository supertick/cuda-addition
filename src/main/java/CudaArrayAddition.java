import java.time.Duration;

import org.bytedeco.cuda.cudart.*;
import org.bytedeco.cuda.global.cudart;
import org.bytedeco.javacpp.FloatPointer;
import org.bytedeco.javacpp.IntPointer;
import org.bytedeco.javacpp.LongPointer;
import org.bytedeco.javacpp.Pointer;
import org.bytedeco.cuda.global.cublas;

import static org.bytedeco.cuda.global.cudart.*;

import java.time.Instant;

import static org.bytedeco.cuda.global.cudart.*;

import org.bytedeco.cuda.global.cudart;
import org.bytedeco.javacpp.DoublePointer;
import org.bytedeco.javacpp.FloatPointer;
import org.bytedeco.javacpp.Pointer;
import org.bytedeco.javacpp.IntPointer;
import org.bytedeco.javacpp.LongPointer;
import org.bytedeco.javacpp.PointerPointer;
import org.bytedeco.javacpp.Pointer;
// import org.bytedeco.cuda.cudart.CUctx_st;
import org.bytedeco.cuda.cudart.CUfunc_st;
import org.bytedeco.cuda.cudart.CUmod_st;

import org.bytedeco.cuda.global.cudart;
import org.bytedeco.cuda.cudart.CUstream_st;
import org.bytedeco.cuda.cudart.CUctx_st;

public class CudaArrayAddition {

    /**
     * Handles CUDA errors by printing an error message and throwing a
     * RuntimeException.
     *
     * @param cudaResult The result code from a CUDA operation.
     * @param msg        A message to include in the error.
     * @return
     */
    public static void handleCudaError(int cudaResult, String msg) {
        if (cudaResult != CUDA_SUCCESS) {
            // Get the error string
            // Pointer errorString = cudaGetErrorString(cudaResult);

            // // Convert Pointer to String using ByteBuffer and StandardCharsets
            // String errMessage = errorString.asByteBuffer().asCharBuffer().toString();

            // Print the error message along with the description
            String errString = String.format("CUDA Error: %d : %s", cudaResult, msg);
            System.err.println(errString);
            throw new RuntimeException(errString);
        }
    }

    public static LongPointer allocAndCopyToGPU(float[] array) {
        // Allocate GPU memory for stateIndices
        long[] arrayDptr = { 0 }; // "pointers" to device memory
        int cudaResult = cuMemAlloc(arrayDptr, array.length * Double.BYTES);
        if (cudaResult != CUDA_SUCCESS) {
            handleCudaError(cudaResult, "Failed to allocate memory for int array");
        }
        FloatPointer nativeArray = new FloatPointer(array); // host->nat
        cudaResult = cuMemcpyHtoD(arrayDptr[0], nativeArray, array.length * Double.BYTES); // nat->dev
        if (cudaResult != CUDA_SUCCESS) {
            handleCudaError(cudaResult, "Failed to copy int array to GPU");
        }
        LongPointer lpStateIndices = new LongPointer(arrayDptr);
        return lpStateIndices;
    }

    public static void main(String[] args) {
        // Initialize CUDA
        cudart.cuInit(0);

        // Get a handle to the device
        IntPointer device = new IntPointer(1);
        int deviceResult = cudart.cuDeviceGet(device, 0); // Get device 0

        if (deviceResult != cudart.CUDA_SUCCESS) {
            System.err.println("Failed to get CUDA device. Error code: " + deviceResult);
            return;
        }

        // Create a CUDA context
        CUctx_st context = new CUctx_st();
        int contextResult = cudart.cuCtxCreate(context, 0, device.get());

        if (contextResult != cudart.CUDA_SUCCESS) {
            System.err.println("Failed to create CUDA context. Error code: " + contextResult);
            return;
        }

        // Array size
        int arraySize = 1000;

        // Allocate and initialize host memory
        float[] hostArrayA = new float[arraySize];
        float[] hostArrayB = new float[arraySize];
        float[] hostArrayC = new float[arraySize];

        for (int i = 0; i < arraySize; i++) {
            hostArrayA[i] = i;
            hostArrayB[i] = i * 2;
        }

        // Allocate device memory once before the loop
        LongPointer deviceArrayA = new LongPointer(1);
        LongPointer deviceArrayB = new LongPointer(1);
        LongPointer deviceArrayC = new LongPointer(1);

        cudart.cuMemAlloc(deviceArrayA, arraySize * Float.BYTES);
        cudart.cuMemAlloc(deviceArrayB, arraySize * Float.BYTES);
        cudart.cuMemAlloc(deviceArrayC, arraySize * Float.BYTES);

        // Copy data from host to device once before the loop
        cudart.cuMemcpyHtoD(deviceArrayA.get(), new FloatPointer(hostArrayA), arraySize * Float.BYTES);
        cudart.cuMemcpyHtoD(deviceArrayB.get(), new FloatPointer(hostArrayB), arraySize * Float.BYTES);

        // Load PTX file
        CUmod_st module = new CUmod_st();
        int result = cudart.cuModuleLoad(module, "add_kernel.ptx");

        if (result != cudart.CUDA_SUCCESS) {
            System.err.println("Failed to load CUDA module. Error code: " + result);
            return;
        }

        // Get function from the module
        CUfunc_st function = new CUfunc_st();
        cudart.cuModuleGetFunction(function, module, "add");

        // Create IntPointer for `n` and set its value to `arraySize`
        IntPointer nPointer = new IntPointer(1).put(arraySize);

        // Kernel parameters
        PointerPointer kernelParams = new PointerPointer(4)
                .put(0, deviceArrayA)
                .put(1, deviceArrayB)
                .put(2, deviceArrayC)
                .put(3, nPointer);

        Instant startTime = Instant.now();

        Instant iterationStart = Instant.now();

        result = cudart.cuLaunchKernel(
                function,
                1, 1, 1, // gridDim
                arraySize, 1, 1, // blockDim
                0, null, // sharedMem and stream
                kernelParams, null // arguments and extra
        );

        if (result != cudart.CUDA_SUCCESS) {
            System.err.println("Kernel launch failed. Error code: " + result);
        }

        result = cudart.cuCtxSynchronize();
        if (result != cudart.CUDA_SUCCESS) {
            System.err
                    .println("Context synchronization failed. Error code: " + result);
        }

        // Copy result from device to host
        FloatPointer nativeMem = new FloatPointer(hostArrayC);
        result = cudart.cuMemcpyDtoH(nativeMem, deviceArrayC.get(), arraySize * Float.BYTES);
        nativeMem.get(hostArrayC);

        if (result != cudart.CUDA_SUCCESS) {
            System.err.println(
                    "Memory copy from device to host failed. Error code: " + result);
        }

        Instant iterationEnd = Instant.now();
        Duration iterationDuration = Duration.between(iterationStart, iterationEnd);
        System.out.println("Iteration  completed in " + iterationDuration.toMillis() + " ms");

        Instant endTime = Instant.now();
        Duration totalDuration = Duration.between(startTime, endTime);
        System.out.println("Test completed in " + totalDuration.toSeconds() + " seconds.");

        // Free device memory after the loop
        cudart.cuMemFree(deviceArrayA.get());
        cudart.cuMemFree(deviceArrayB.get());
        cudart.cuMemFree(deviceArrayC.get());

        for (int i = 0; i < arraySize; i++) {
            System.out.println(hostArrayC[i]);
        }
    }
}
