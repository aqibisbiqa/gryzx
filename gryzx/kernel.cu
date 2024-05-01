#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <chrono>
#include <iostream>
#include <algorithm>

const int NVERTS     = 5;    // Number of vertices in the graph
const int RANDSEED   = 227;  // Seed for randomisation
const double DENSITY = 0.7;  // Density of the random graph (0.0 = no connections, 1.0 = fully connected)
const bool doDisplay = true; // Toggles whether to display the adjacency matrices

// STEP 1:
// Fuse v_b into v_a...
__global__
void kFuse(int n, short* e, short* eBuffer, int nVerts, int v_a, int v_b) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i > n-1) return;

    int x = i % nVerts;
    int y = i / nVerts;
    
    int xDiff = v_b - v_a;
    int yDiff = xDiff * nVerts;

    // Column:
    if (x == v_a) e[i] = (eBuffer[i] + eBuffer[i + xDiff]) % 2;

    // Row:
    if (y == v_a) e[i] = (eBuffer[i] + eBuffer[i + yDiff]) % 2;
}

// STEP 2:
// Set v_b row and column to zero...
__global__
void kFuse2(int n, short* e, short* eBuffer, int nVerts, int v_a, int v_b) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i > n - 1) return;

    int x = i % nVerts;
    int y = i / nVerts;

    // Column:
    if (x == v_b) e[i] = 0;

    // Row:
    if (y == v_b) e[i] = 0;

    // If pointing to self:
    if (x == v_a && y == v_a) e[i] = 0;
}

void simp_fuse(int i, int a, short* e) {
    e[i] = a * e[i];
}

/*void genGraph(short* e, int nVerts) {
    //TEMP:
    short connections[] = {
        0,1,0,0,0,
        1,0,1,0,0,
        0,1,0,1,1,
        0,0,1,0,1,
        0,0,1,1,0
    };

    for (int i = 0; i < nVerts * nVerts; ++i) {
        e[i] = connections[i];
    }
}*/

int randBit(double prob) { // Returns 1 with probability prob, and 0 with probability (1-prob)
    int r = rand();
    if (r < RAND_MAX * prob) return 1;
    return 0;
}

void genRandGraph(short* e, double density) {
    // Initialise leading diagonal to zero...
    for (int y = 0; y < NVERTS; ++y) {
        e[(y * NVERTS) + y] = 0;
    }

    // Randomly populate graph (symmetrically)...
    for (int y = 0; y < NVERTS; ++y) {
        for (int x = y+1; x < NVERTS; ++x) {
            int r = randBit(density);
            e[(y * NVERTS) + x] = r;
            e[(x * NVERTS) + y] = r;
        }
    }
}

void copyGraph(short* eFrom, short* eTo, int nVerts) {
    for (int i = 0; i < nVerts * nVerts; ++i) {
        eTo[i] = eFrom[i];
    }
}

void printGraph(short* e, int nVerts) {
    std::cout << "\n\n";

    for (int y = 0; y < nVerts; ++y) {
        for (int x = 0; x < nVerts; ++x) {
            std::cout << e[y * nVerts + x] << " ";
        }
        std::cout << "\n";
    }

    std::cout << "\n\n";
}

int main(void) {
    srand(RANDSEED); // initialize seed for randomisation

    int nVerts = NVERTS;
    int nCells = nVerts * nVerts;
    short* e, * d_e, * e_buffer, * d_e_buffer;
    e = (short*)malloc(nCells * sizeof(short));
    e_buffer = (short*)malloc(nCells * sizeof(short));

    cudaMalloc(&d_e, nCells * sizeof(short));
    cudaMalloc(&d_e_buffer, nCells * sizeof(short));

    //genGraph(e, nVerts);
    genRandGraph(e, DENSITY);
    copyGraph(e, e_buffer, nVerts);
    if (doDisplay) printGraph(e, nVerts);

    cudaMemcpy(d_e, e, nCells * sizeof(short), cudaMemcpyHostToDevice);
    cudaMemcpy(d_e_buffer, e_buffer, nCells * sizeof(short), cudaMemcpyHostToDevice);

    auto tStart = std::chrono::steady_clock::now();

    // Run the GPU kernels...
    kFuse << <(nCells + 255) / 256, 256 >> > (nCells, d_e, d_e_buffer, nVerts, 2, 3);
    kFuse2 << <(nCells + 255) / 256, 256 >> > (nCells, d_e, d_e_buffer, nVerts, 2, 3);
    cudaDeviceSynchronize();
    auto tMid = std::chrono::steady_clock::now();
    cudaMemcpy(e, d_e, nCells * sizeof(short), cudaMemcpyDeviceToHost);

    auto tEnd = std::chrono::steady_clock::now();
    auto tDiff_kern = std::chrono::duration_cast<std::chrono::microseconds>(tMid - tStart).count();
    auto tDiff_tot  = std::chrono::duration_cast<std::chrono::microseconds>(tEnd - tStart).count();
    std::cout << "gpu: " << tDiff_tot << " us  (" << tDiff_kern << " us excluding memory transfer)\n";

    if (doDisplay) printGraph(e, nVerts);

    //cudaMemcpy(y, d_y, N * sizeof(float), cudaMemcpyDeviceToHost);

    //--
    // CPU approach...

    tStart = std::chrono::steady_clock::now();

    for (int i = 0; i < nCells; ++i) {
        simp_fuse(i, 2, e);
    }

    tEnd = std::chrono::steady_clock::now();
    tDiff_tot = std::chrono::duration_cast<std::chrono::microseconds>(tEnd - tStart).count();
    std::cout << "cpu: " << tDiff_tot << " us\n";

    //--

    cudaFree(d_e);
    free(e);
}