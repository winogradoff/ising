#include <cstdio>
#include <cuda.h>

// cuRAND
#include <curand.h>
#include <curand_kernel.h>

#include "grid.h"

extern "C"

dim3 blocks(8, 8, 8);
dim3 threads(8, 8, 8);

__device__
int getIndex(int xSize, int ySize, int zSize, int i, int j, int k)
{
    return (i * ySize + j) * zSize + k;
}

__global__
void kernelInitRandomStates(curandState *randomStates, int xSize, int ySize, int zSize)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int idy = blockIdx.y * blockDim.y + threadIdx.y;
    int idz = blockIdx.z * blockDim.z + threadIdx.z;
    int offsetx = gridDim.x * blockDim.x;
    int offsety = gridDim.y * blockDim.y;
    int offsetz = gridDim.z * blockDim.z;

    int index = getIndex(offsetx, offsety, offsetz, idx, idy, idz);
    curand_init(clock64(), index, 0, &(randomStates[index]));
}

__global__
void kernelInitGrid(
    BYTE *data, curandState *randomStates,
    int xSize, int ySize, int zSize
)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int idy = blockIdx.y * blockDim.y + threadIdx.y;
    int idz = blockIdx.z * blockDim.z + threadIdx.z;
    int offsetx = gridDim.x * blockDim.x;
    int offsety = gridDim.y * blockDim.y;
    int offsetz = gridDim.z * blockDim.z;

    curandState *rndState = &(randomStates[getIndex(offsetx, offsety, offsetz, idx, idy, idz)]);

    for (int i = idx; i < xSize; i += offsetx)
    {
        for (int j = idy; j < ySize; j += offsety)
        {
            for (int k = idz; k < zSize; k += offsetz)
            {
                BYTE value = curand_uniform(rndState) < 0.5 ? 0 : 2;
                data[getIndex(xSize, ySize, zSize, i, j, k)] = value;
            }
        }
    }
}

__device__
double gridDistantion(int i, int j, int k, int x, int y, int z)
{
    return sqrt(double((i - x) * (i - x) + (j - y) * (j - y) + (k - z) * (k - z)));
}

__device__
double gridInteractionPotential(double interactionEnergy, double r)
{
    return interactionEnergy / (r * r);
}

__global__
void kernelAlgorithm(
    BYTE *data, curandState *randomStates,
    DimensionEnum dimension, int xSize, int ySize, int zSize,
    int interactionEnergy, double externalField, int interactionRadius, double temperature,
    bool even
)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int idy = blockIdx.y * blockDim.y + threadIdx.y;
    int idz = blockIdx.z * blockDim.z + threadIdx.z;
    int offsetx = gridDim.x * blockDim.x;
    int offsety = gridDim.y * blockDim.y;
    int offsetz = gridDim.z * blockDim.z;

    curandState *rndState = &(randomStates[getIndex(offsetx, offsety, offsetz, idx, idy, idz)]);

    for (int i = idx; i < xSize; i += offsetx)
    {
        for (int j = idy; j < ySize; j += offsety)
        {
            for (int k = idz; k < zSize; k += offsetz)
            {
                int index = getIndex(xSize, ySize, zSize, i, j, k);

                if (even && index % 2 != 0) continue;
                if (!even && index % 2 == 0) continue;

                int radiusX = interactionRadius;
                int radiusY = interactionRadius;
                int radiusZ = interactionRadius;

                switch (dimension)
                {
                    case DIM_1: radiusY = radiusZ = 0; break;
                    case DIM_2: radiusZ = 0; break;
                    case DIM_3: break;
                }

                double gridSpinEnergy = externalField;
                for (int x = i - radiusX; x <= i + radiusX; x++)
                {
                    for (int y = j - radiusY; y <= j + radiusY; y++)
                    {
                        for (int z = k - radiusZ; z <= k + radiusZ; z++)
                        {
                            int xx = (xSize + x) % xSize;
                            int yy = (ySize + y) % ySize;
                            int zz = (zSize + z) % zSize;

                            if (xx == i && yy == j && zz == k) continue;

                            double dist = gridDistantion(i, j, k, xx, yy, zz);

                            if (dist <= interactionRadius) {
                                gridSpinEnergy += (data[getIndex(xSize, ySize, zSize, xx, yy, zz)] - 1)
                                                  * gridInteractionPotential(interactionEnergy, dist);
                            }
                        }
                    }
                }

                double expValue = exp(gridSpinEnergy / temperature);
                double probplus  = expValue;
                double probminus = 1.0 / expValue;
                double probability = probplus / (probplus + probminus);

                if (curand_uniform(rndState) > probability)
                {
                    data[index] = 0;
                }
                else
                {
                    data[index] = 2;
                }
            }
        }
    }
}

void cudaInitGrid(Grid *g)
{
    int x = blocks.x * threads.x;
    int y = blocks.y * threads.y;
    int z = blocks.z * threads.z;
    cudaMalloc((void **)& (g->randomStates), sizeof(curandState) * x * y * z);

    int dataSize = g->xSize * g->ySize * g->zSize;
    cudaMalloc((void **)& (g->deviceMatrix), sizeof(BYTE) * dataSize);
    kernelInitRandomStates<<<blocks, threads>>>(g->randomStates, g->xSize, g->ySize, g->zSize);
    kernelInitGrid<<<blocks, threads>>>(g->deviceMatrix, g->randomStates, g->xSize, g->ySize, g->zSize);
}

void cudaFreeGrid(Grid *g)
{
    if (g->randomStates != NULL) cudaFree(g->randomStates);
    if (g->deviceMatrix != NULL) cudaFree(g->deviceMatrix);
}

void cudaUpdateTempMatrix(Grid *g)
{
    int dataSize = g->xSize * g->ySize * g->zSize;
    cudaMemcpy(g->hostMatrix, g->deviceMatrix, sizeof(BYTE) * dataSize, cudaMemcpyDeviceToHost);
}

void cudaAlgorithmStep(Grid *g, int algorithmSteps)
{
    for (int i = 0; i < algorithmSteps; i++)
    {
        kernelAlgorithm<<<blocks, threads>>>(
            g->deviceMatrix, g->randomStates,
            g->dimension, g->xSize, g->ySize, g->zSize,
            g->interactionEnergy, g->externalField, g->interactionRadius, g->temperature, false
        );

        kernelAlgorithm<<<blocks, threads>>>(
            g->deviceMatrix, g->randomStates,
            g->dimension, g->xSize, g->ySize, g->zSize,
            g->interactionEnergy, g->externalField, g->interactionRadius, g->temperature, true
        );
    }
}
