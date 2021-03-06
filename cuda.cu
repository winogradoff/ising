#include <cstdio>

#include <cuda.h>

// cuRAND
#include <curand.h>
#include <curand_kernel.h>

// Thrust
#include <thrust/device_vector.h>
#include <thrust/reduce.h>

#include "types.h"
#include "grid.h"

dim3 blocks_1d(1, 1, 1);
dim3 threads_1d(1024, 1, 1);

dim3 blocks_2d(16, 16, 1);
dim3 threads_2d(32, 32, 1);

dim3 blocks_3d(4, 8, 8);
dim3 threads_3d(16, 8, 8);

dim3 blocks, threads;

__constant__ DimensionEnum dimension;
__constant__ uint xSize;
__constant__ uint ySize;
__constant__ uint zSize;
__constant__ int interactionEnergy;
__constant__ double externalField;
__constant__ int interactionRadius;
__constant__ double temperature;
__constant__ int percentOfCube;
__constant__ int percentOfNonmagnetic;

double* tempMatrix;

__device__
uint linearIndex(uint iSize, uint jSize, uint kSize, uint i, uint j, uint k)
{
    return (i * jSize + j) * kSize + k;
}

__device__
uint gridIndex(uint i, uint j, uint k)
{
    return linearIndex(xSize, ySize, zSize, i, j, k);
}

__device__
double gridDistantion(uint i, uint j, uint k, uint x, uint y, uint z)
{
    return sqrt(double((i - x) * (i - x) + (j - y) * (j - y) + (k - z) * (k - z)));
}

__device__
double gridInteractionPotential(double r)
{
    return interactionEnergy / (r * r);
}

// float -> half float
__device__
short floatToHalf(float value)
{
    short fltInt16;
    int fltInt32;
    memcpy(&fltInt32, &value, sizeof(float));
    fltInt16 = ((fltInt32 & 0x7fffffff) >> 13) - (0x38000000 >> 13);
    fltInt16 |= ((fltInt32 & 0x80000000) >> 16);
    return fltInt16;
}

__device__
VBOVertex makeVertex(float x, float y, float z, uchar r, uchar g, uchar b, uchar a)
{
    return VBOVertex{ x, y, z, r, g, b, a };
    //    return VBOVertex{floatToHalf(x), floatToHalf(y), floatToHalf(z), r, g, b, a};
}

__global__
void kernelInitRandomStates(curandState* randomStates)
{
    uint idx = blockIdx.x * blockDim.x + threadIdx.x;
    uint idy = blockIdx.y * blockDim.y + threadIdx.y;
    uint idz = blockIdx.z * blockDim.z + threadIdx.z;
    uint offsetx = gridDim.x * blockDim.x;
    uint offsety = gridDim.y * blockDim.y;
    uint offsetz = gridDim.z * blockDim.z;

    uint index = linearIndex(offsetx, offsety, offsetz, idx, idy, idz);
    curand_init(clock64(), index, 0, &(randomStates[index]));
}

__global__
void kernelInitGrid(uchar* data, curandState* randomStates)
{
    uint idx = blockIdx.x * blockDim.x + threadIdx.x;
    uint idy = blockIdx.y * blockDim.y + threadIdx.y;
    uint idz = blockIdx.z * blockDim.z + threadIdx.z;
    uint offsetx = gridDim.x * blockDim.x;
    uint offsety = gridDim.y * blockDim.y;
    uint offsetz = gridDim.z * blockDim.z;

    curandState* rndState = &(randomStates[linearIndex(offsetx, offsety, offsetz, idx, idy, idz)]);

    for (uint i = idx; i < xSize; i += offsetx)
    {
        for (uint j = idy; j < ySize; j += offsety)
        {
            for (uint k = idz; k < zSize; k += offsetz)
            {
                uint index = gridIndex(i, j, k);
                if (curand_uniform(rndState) < percentOfNonmagnetic / 100.0)
                {
                    data[index] = 1;
                }
                else
                {
                    uchar value = curand_uniform(rndState) < 0.5 ? 0 : 2;
                    data[index] = value;
                }
            }
        }
    }
}

__global__
void kernelAlgorithm(
    uchar* data, curandState* randomStates, int iterx, int itery, int iterz)
{
    uint idx = blockIdx.x * blockDim.x + threadIdx.x;
    uint idy = blockIdx.y * blockDim.y + threadIdx.y;
    uint idz = blockIdx.z * blockDim.z + threadIdx.z;
    uint offsetx = gridDim.x * blockDim.x;
    uint offsety = gridDim.y * blockDim.y;
    uint offsetz = gridDim.z * blockDim.z;

    uint rndIndex = linearIndex(offsetx, offsety, offsetz, idx, idy, idz);
    curandState* rndState = &(randomStates[rndIndex]);

    uint gridOffsetx = offsetx * (interactionRadius + 1);
    uint gridOffsety = offsety * (interactionRadius + 1);
    uint gridOffsetz = offsetz * (interactionRadius + 1);

    int index, radiusX, radiusY, radiusZ;
    uchar spinValue;
    uint xx, yy, zz;
    double gridSpinEnergy, dist, expValue, probability;

    radiusX = radiusY = radiusZ = interactionRadius;
    switch (dimension)
    {
        case DIM_1:
            radiusY = radiusZ = 0;
            break;
        case DIM_2:
            radiusZ = 0;
            break;
    }

    for (int i = idx + idx * interactionRadius + iterx; i < xSize; i += gridOffsetx)
    {
        for (int j = idy + idy * interactionRadius + itery; j < ySize; j += gridOffsety)
        {
            for (int k = idz + idz * interactionRadius + iterz; k < zSize; k += gridOffsetz)
            {
                index = gridIndex(i, j, k);
                spinValue = data[index] - 1;

                // Пропустить, если немагнитная частица
                if (spinValue == 0) continue;

                gridSpinEnergy = externalField;

                for (int x = i - radiusX; x <= i + radiusX; x++)
                {
                    for (int y = j - radiusY; y <= j + radiusY; y++)
                    {
                        for (int z = k - radiusZ; z <= k + radiusZ; z++)
                        {
                            xx = (xSize + x) % xSize;
                            yy = (ySize + y) % ySize;
                            zz = (zSize + z) % zSize;

                            if (xx == i && yy == j && zz == k)
                                continue;

                            dist = gridDistantion(i, j, k, x, y, z);

                            if (dist <= interactionRadius)
                            {
                                gridSpinEnergy += (data[gridIndex(xx, yy, zz)] - 1)
                                    * gridInteractionPotential(dist);
                            }
                        }
                    }
                }

                expValue = exp(gridSpinEnergy / temperature);
                probability = expValue / (expValue + 1.0 / expValue);

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

__global__
void kernelMagnetization(uchar* data, double* result)
{
    uint idx = blockIdx.x * blockDim.x + threadIdx.x;
    uint idy = blockIdx.y * blockDim.y + threadIdx.y;
    uint idz = blockIdx.z * blockDim.z + threadIdx.z;
    uint offsetx = gridDim.x * blockDim.x;
    uint offsety = gridDim.y * blockDim.y;
    uint offsetz = gridDim.z * blockDim.z;

    for (uint i = idx; i < xSize; i += offsetx)
    {
        for (uint j = idy; j < ySize; j += offsety)
        {
            for (uint k = idz; k < zSize; k += offsetz)
            {
                int index = gridIndex(i, j, k);
                result[index] = ((double)data[index]) - 1.0;
            }
        }
    }
}

__global__
void kernelEnergy(uchar* data, double* result)
{
    uint idx = blockIdx.x * blockDim.x + threadIdx.x;
    uint idy = blockIdx.y * blockDim.y + threadIdx.y;
    uint idz = blockIdx.z * blockDim.z + threadIdx.z;
    uint offsetx = gridDim.x * blockDim.x;
    uint offsety = gridDim.y * blockDim.y;
    uint offsetz = gridDim.z * blockDim.z;

    int index, radiusX, radiusY, radiusZ;
    uint xx, yy, zz;
    double spinValue, dist, energy;

    radiusX = radiusY = radiusZ = interactionRadius;
    switch (dimension)
    {
        case DIM_1:
            radiusY = radiusZ = 0;
            break;
        case DIM_2:
            radiusZ = 0;
            break;
    }

    for (int i = idx; i < xSize; i += offsetx)
    {
        for (int j = idy; j < ySize; j += offsety)
        {
            for (int k = idz; k < zSize; k += offsetz)
            {
                index = gridIndex(i, j, k);

                energy = 0.0;
                spinValue = ((double)data[index]) - 1.0;

                if (spinValue == 0)
                {
                    result[index] = 0;
                    continue;
                }

                for (int x = i; x <= i + radiusX; x++)
                {
                    for (int y = j; y <= j + radiusY; y++)
                    {
                        for (int z = k; z <= k + radiusZ; z++)
                        {
                            xx = (xSize + x) % xSize;
                            yy = (ySize + y) % ySize;
                            zz = (zSize + z) % zSize;

                            if (xx == i && yy == j && zz == k)
                                continue;

                            dist = gridDistantion(i, j, k, x, y, z);

                            if (dist <= interactionRadius)
                            {
                                energy += (((double)data[gridIndex(xx, yy, zz)]) - 1)
                                    * gridInteractionPotential(dist);
                            }
                        }
                    }
                }

                result[index] = -spinValue * (energy * interactionEnergy + externalField);
            }
        }
    }
}

__global__
void kernelInitVBO(uchar* data, VBOVertex* vertices, uint* indices)
{
    uint idx = blockIdx.x * blockDim.x + threadIdx.x;
    uint idy = blockIdx.y * blockDim.y + threadIdx.y;
    uint idz = blockIdx.z * blockDim.z + threadIdx.z;
    uint offsetx = gridDim.x * blockDim.x;
    uint offsety = gridDim.y * blockDim.y;
    uint offsetz = gridDim.z * blockDim.z;

    float gridSize = 5.0f;
    uint maxSize = max(xSize, max(ySize, zSize));
    float cubeSize = (gridSize / maxSize) * (0.01f * percentOfCube);
    float cubeSpace = (gridSize / maxSize) * (0.01f * (100 - percentOfCube));

    float lengthX = xSize * cubeSize + (xSize - 1) * cubeSpace;
    float lengthY = ySize * cubeSize + (ySize - 1) * cubeSpace;
    float lengthZ = zSize * cubeSize + (zSize - 1) * cubeSpace;

    for (uint i = idx; i < xSize; i += offsetx)
    {
        for (uint j = idy; j < ySize; j += offsety)
        {
            for (uint k = idz; k < zSize; k += offsetz)
            {
                uint index = gridIndex(i, j, k);
                uchar r, g, b, a;

                switch (data[index])
                {
                    case 0:
                        r = 0;
                        g = 255;
                        b = 0;
                        a = 255;
                        break;
                    case 1:
                        r = 255;
                        g = 255;
                        b = 255;
                        a = 255;
                        break;
                    case 2:
                        r = 255;
                        g = 0;
                        b = 0;
                        a = 255;
                        break;
                }

                float x = i * (cubeSize + cubeSpace) + 0.5f * (cubeSize - lengthX) - 0.5f * cubeSize;
                float y = j * (cubeSize + cubeSpace) + 0.5f * (cubeSize - lengthY) - 0.5f * cubeSize;
                float z = k * (cubeSize + cubeSpace) + 0.5f * (cubeSize - lengthZ) - 0.5f * cubeSize;

                // 8 вершин куба
                uint vpos = index * 8;

                vertices[vpos + 0] = makeVertex(x, y, z, r, g, b, a);
                vertices[vpos + 1] = makeVertex(x, y, z + cubeSize, r, g, b, a);
                vertices[vpos + 2] = makeVertex(x, y + cubeSize, z, r, g, b, a);
                vertices[vpos + 3] = makeVertex(x, y + cubeSize, z + cubeSize, r, g, b, a);
                vertices[vpos + 4] = makeVertex(x + cubeSize, y, z, r, g, b, a);
                vertices[vpos + 5] = makeVertex(x + cubeSize, y, z + cubeSize, r, g, b, a);
                vertices[vpos + 6] = makeVertex(x + cubeSize, y + cubeSize, z, r, g, b, a);
                vertices[vpos + 7]
                    = makeVertex(x + cubeSize, y + cubeSize, z + cubeSize, r, g, b, a);

                // 24 индекса для рисования GL_QUADS
                uint ipos = index * 24;

                // Перед
                indices[ipos++] = vpos + 0;
                indices[ipos++] = vpos + 4;
                indices[ipos++] = vpos + 5;
                indices[ipos++] = vpos + 1;

                // Зад
                indices[ipos++] = vpos + 2;
                indices[ipos++] = vpos + 3;
                indices[ipos++] = vpos + 7;
                indices[ipos++] = vpos + 6;

                // Верх
                indices[ipos++] = vpos + 1;
                indices[ipos++] = vpos + 5;
                indices[ipos++] = vpos + 7;
                indices[ipos++] = vpos + 3;

                // Низ
                indices[ipos++] = vpos + 0;
                indices[ipos++] = vpos + 2;
                indices[ipos++] = vpos + 6;
                indices[ipos++] = vpos + 4;

                // Лево
                indices[ipos++] = vpos + 0;
                indices[ipos++] = vpos + 1;
                indices[ipos++] = vpos + 3;
                indices[ipos++] = vpos + 2;

                // Право
                indices[ipos++] = vpos + 4;
                indices[ipos++] = vpos + 6;
                indices[ipos++] = vpos + 7;
                indices[ipos++] = vpos + 5;
            }
        }
    }
}

__global__ void kernelUpdateVBO(uchar* data, VBOVertex* verts)
{
    uint idx = blockIdx.x * blockDim.x + threadIdx.x;
    uint idy = blockIdx.y * blockDim.y + threadIdx.y;
    uint idz = blockIdx.z * blockDim.z + threadIdx.z;
    uint offsetx = gridDim.x * blockDim.x;
    uint offsety = gridDim.y * blockDim.y;
    uint offsetz = gridDim.z * blockDim.z;

    for (uint i = idx; i < xSize; i += offsetx)
    {
        for (uint j = idy; j < ySize; j += offsety)
        {
            for (uint k = idz; k < zSize; k += offsetz)
            {
                uint index = gridIndex(i, j, k);
                uchar r, g, b, a;

                switch (data[index])
                {
                    case 0:
                        r = 0;
                        g = 255;
                        b = 0;
                        a = 0;
                        break;
                    case 1:
                        r = 255;
                        g = 255;
                        b = 255;
                        a = 0;
                        break;
                    case 2:
                        r = 255;
                        g = 0;
                        b = 0;
                        a = 0;
                        break;
                }

                for (uint v = index * 8; v < index * 8 + 8; v++)
                {
                    verts[v].r = r;
                    verts[v].g = g;
                    verts[v].b = b;
                    verts[v].a = a;
                }
            }
        }
    }
}

void cudaInitGrid(Grid* g)
{
    cudaMemcpyToSymbol((const void*)&dimension, &(g->dimension), sizeof(DimensionEnum));
    cudaMemcpyToSymbol((const void*)&xSize, &(g->xSize), sizeof(int));
    cudaMemcpyToSymbol((const void*)&ySize, &(g->ySize), sizeof(int));
    cudaMemcpyToSymbol((const void*)&zSize, &(g->zSize), sizeof(int));
    cudaMemcpyToSymbol((const void*)&interactionEnergy, &(g->interactionEnergy), sizeof(int));
    cudaMemcpyToSymbol((const void*)&externalField, &(g->externalField), sizeof(double));
    cudaMemcpyToSymbol((const void*)&interactionRadius, &(g->interactionRadius), sizeof(int));
    cudaMemcpyToSymbol((const void*)&temperature, &(g->temperature), sizeof(double));
    cudaMemcpyToSymbol((const void*)&percentOfCube, &(g->percentOfCube), sizeof(int));
    cudaMemcpyToSymbol((const void*)&percentOfNonmagnetic, &(g->percentOfNonmagnetic), sizeof(int));

    switch (g->dimension)
    {
        case DIM_1:
            blocks = blocks_1d;
            threads = threads_1d;
            break;
        case DIM_2:
            blocks = blocks_2d;
            threads = threads_2d;
            break;
        case DIM_3:
            blocks = blocks_3d;
            threads = threads_3d;
            break;
    }

    uint cudaSize = blocks.x * blocks.y * blocks.z * threads.x * threads.y * threads.z;
    cudaMalloc((void**)&(g->randomStates), sizeof(curandState) * cudaSize);
    kernelInitRandomStates<<<blocks, threads>>>(g->randomStates);

    uint dataSize = g->xSize * g->ySize * g->zSize;
    cudaMalloc((void**)&(g->deviceMatrix), sizeof(uchar) * dataSize);
    cudaMalloc((void**)&tempMatrix, sizeof(double) * dataSize);
    kernelInitGrid<<<blocks, threads>>>(g->deviceMatrix, g->randomStates);
}

void cudaSetParams(Grid* g)
{
    cudaMemcpyToSymbol((const void*)&interactionEnergy, &(g->interactionEnergy), sizeof(int));
    cudaMemcpyToSymbol((const void*)&externalField, &(g->externalField), sizeof(double));
    cudaMemcpyToSymbol((const void*)&interactionRadius, &(g->interactionRadius), sizeof(int));
    cudaMemcpyToSymbol((const void*)&temperature, &(g->temperature), sizeof(double));
    cudaMemcpyToSymbol((const void*)&percentOfCube, &(g->percentOfCube), sizeof(int));
}

void cudaFreeGrid(Grid* g)
{
    if (g->randomStates != NULL)
    {
        cudaFree(g->randomStates);
        g->randomStates = NULL;
    }

    if (g->deviceMatrix != NULL)
    {
        cudaFree(g->deviceMatrix);
        g->deviceMatrix = NULL;
    }

    if (tempMatrix != NULL)
    {
        cudaFree(tempMatrix);
        tempMatrix = NULL;
    }
}

void cudaAlgorithmStep(Grid* g, uint algorithmSteps)
{
    for (uint i = 0; i < algorithmSteps; i++)
    {
        for (int iterx = 0; iterx <= g->interactionRadius; iterx++)
        {
            for (int itery = 0; itery <= g->interactionRadius; itery++)
            {
                for (int iterz = 0; iterz <= g->interactionRadius; iterz++)
                {
                    kernelAlgorithm<<<blocks, threads>>>(
                        g->deviceMatrix, g->randomStates, iterx, itery, iterz);
                }
            }
        }
    }
}

double cudaMagnetization(Grid* g)
{
    kernelMagnetization<<<blocks, threads>>>(g->deviceMatrix, tempMatrix);
    double sum = 0.0;
    uint dataSize = g->xSize * g->ySize * g->zSize;
    thrust::device_ptr<double> ptr(tempMatrix);
    sum = thrust::reduce(ptr, ptr + dataSize, sum, thrust::plus<double>());
    return sum;
}

double cudaEnergy(Grid* g)
{
    kernelEnergy<<<blocks, threads>>>(g->deviceMatrix, tempMatrix);
    double sum = 0.0;
    uint dataSize = g->xSize * g->ySize * g->zSize;
    thrust::device_ptr<double> ptr(tempMatrix);
    sum = thrust::reduce(ptr, ptr + dataSize, sum, thrust::plus<double>());
    return sum;
}

void cudaInitVBO(Grid* g, struct cudaGraphicsResource** cudaVertexResource,
    struct cudaGraphicsResource** cudaIndexResource)
{
    VBOVertex* vertices;
    uint* indexes;

    size_t num_bytes;

    cudaGraphicsMapResources(1, cudaVertexResource, 0);
    cudaGraphicsResourceGetMappedPointer((void**)&vertices, &num_bytes, *cudaVertexResource);
    cudaGraphicsMapResources(1, cudaIndexResource, 0);
    cudaGraphicsResourceGetMappedPointer((void**)&indexes, &num_bytes, *cudaIndexResource);

    kernelInitVBO<<<blocks, threads>>>(g->deviceMatrix, vertices, indexes);

    cudaGraphicsUnmapResources(1, cudaVertexResource, 0);
    cudaGraphicsUnmapResources(1, cudaIndexResource, 0);
}

void cudaUpdateVBO(Grid* g, struct cudaGraphicsResource** cudaVertexResource)
{
    VBOVertex* vertices;
    size_t num_bytes;

    cudaGraphicsMapResources(1, cudaVertexResource, 0);
    cudaGraphicsResourceGetMappedPointer((void**)&vertices, &num_bytes, *cudaVertexResource);

    kernelUpdateVBO<<<blocks, threads>>>(g->deviceMatrix, vertices);

    cudaGraphicsUnmapResources(1, cudaVertexResource, 0);
}
