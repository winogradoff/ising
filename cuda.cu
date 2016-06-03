#include <cstdio>
#include <cuda.h>

// cuRAND
#include <curand.h>
#include <curand_kernel.h>

#include "types.h"
#include "grid.h"

extern "C"

//dim3 blocks(8, 8, 8);
//dim3 threads(8, 8, 8);

dim3 threads(8, 8, 8);

__device__
uint getIndex(uint xSize, uint ySize, uint zSize, uint i, uint j, uint k)
{    
    return (i * ySize + j) * zSize + k;
}

__device__
double gridDistantion(uint i, uint j, uint k, uint x, uint y, uint z)
{
    return sqrt(double((i - x) * (i - x) + (j - y) * (j - y) + (k - z) * (k - z)));
}

__device__
double gridInteractionPotential(double interactionEnergy, double r)
{
    return interactionEnergy / (r * r);
}

__global__
void kernelInitRandomStates(curandState *randomStates, uint xSize, uint ySize, uint zSize)
{
    uint idx = blockIdx.x * blockDim.x + threadIdx.x;
    uint idy = blockIdx.y * blockDim.y + threadIdx.y;
    uint idz = blockIdx.z * blockDim.z + threadIdx.z;
    uint offsetx = gridDim.x * blockDim.x;
    uint offsety = gridDim.y * blockDim.y;
    uint offsetz = gridDim.z * blockDim.z;

    uint index = getIndex(offsetx, offsety, offsetz, idx, idy, idz);
    curand_init(clock64(), index, 0, &(randomStates[index]));
}

__global__
void kernelInitGrid(
    uchar *data, curandState *randomStates,
    uint xSize, uint ySize, uint zSize
)
{
    uint idx = blockIdx.x * blockDim.x + threadIdx.x;
    uint idy = blockIdx.y * blockDim.y + threadIdx.y;
    uint idz = blockIdx.z * blockDim.z + threadIdx.z;
    uint offsetx = gridDim.x * blockDim.x;
    uint offsety = gridDim.y * blockDim.y;
    uint offsetz = gridDim.z * blockDim.z;

    curandState *rndState = &(randomStates[getIndex(offsetx, offsety, offsetz, idx, idy, idz)]);

    for (uint i = idx; i < xSize; i += offsetx)
    {
        for (uint j = idy; j < ySize; j += offsety)
        {
            for (uint k = idz; k < zSize; k += offsetz)
            {
                uchar value = curand_uniform(rndState) < 0.5 ? 0 : 2;
                data[getIndex(xSize, ySize, zSize, i, j, k)] = value;
            }
        }
    }
}

__global__
void kernelAlgorithm(
    uchar *data, curandState *randomStates,
    DimensionEnum dimension, uint xSize, uint ySize, uint zSize,
    int interactionEnergy, double externalField, int interactionRadius, double temperature,
    bool even
)
{
    uint idx = blockIdx.x * blockDim.x + threadIdx.x;
    uint idy = blockIdx.y * blockDim.y + threadIdx.y;
    uint idz = blockIdx.z * blockDim.z + threadIdx.z;
    uint offsetx = gridDim.x * blockDim.x;
    uint offsety = gridDim.y * blockDim.y;
    uint offsetz = gridDim.z * blockDim.z;

    curandState *rndState = &(randomStates[getIndex(offsetx, offsety, offsetz, idx, idy, idz)]);

    for (uint i = idx; i < xSize; i += offsetx)
    {
        for (uint j = idy; j < ySize; j += offsety)
        {
            for (uint k = idz; k < zSize; k += offsetz)
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
                            uint xx = (xSize + x) % xSize;
                            uint yy = (ySize + y) % ySize;
                            uint zz = (zSize + z) % zSize;

                            if (xx == i && yy == j && zz == k) continue;

                            double dist = gridDistantion(i, j, k, xx, yy, zz);

                            if (dist <= interactionRadius)
                            {
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

__global__
void kernelInitVBO(VBOVertex *verts, uchar *data, uint xSize, uint ySize, uint zSize, float percentOfCube)
{
    uint idx = blockIdx.x * blockDim.x + threadIdx.x;
    uint idy = blockIdx.y * blockDim.y + threadIdx.y;
    uint idz = blockIdx.z * blockDim.z + threadIdx.z;
    uint offsetx = gridDim.x * blockDim.x;
    uint offsety = gridDim.y * blockDim.y;
    uint offsetz = gridDim.z * blockDim.z;

    float gridSize = 5.0f;

    // TODO: максимальный count
    float cubeSize = (gridSize / xSize) * percentOfCube;
    float cubeSpace = (gridSize / xSize) * (1.0f - percentOfCube);

    float lengthX = xSize * cubeSize + (xSize - 1) * cubeSpace;
    float lengthY = ySize * cubeSize + (ySize - 1) * cubeSpace;
    float lengthZ = zSize * cubeSize + (zSize - 1) * cubeSpace;

    for (uint i = idx; i < xSize; i += offsetx)
    {
        for (uint j = idy; j < ySize; j += offsety)
        {
            for (uint k = idz; k < zSize; k += offsetz)
            {
                uint index = getIndex(xSize, ySize, zSize, i, j, k);
                uchar r, g, b, a;

                switch (data[index])
                {
                    case 0: r = 0;   g = 255; b = 0;   a = 255; break;
                    case 1: r = 255; g = 255; b = 255; a = 255; break;
                    case 2: r = 255; g = 0;   b = 0;   a = 255; break;
                }

                float x = i * (cubeSize + cubeSpace) + 0.5f * (cubeSize - lengthX) - 0.5f * cubeSize;
                float y = j * (cubeSize + cubeSpace) + 0.5f * (cubeSize - lengthY) - 0.5f * cubeSize;
                float z = k * (cubeSize + cubeSpace) + 0.5f * (cubeSize - lengthZ) - 0.5f * cubeSize;

                uint v = index * 24;

                // Перед
                verts[v++] = VBOVertex{x,            y,            z           , r, g, b, a};
                verts[v++] = VBOVertex{x + cubeSize, y,            z           , r, g, b, a};
                verts[v++] = VBOVertex{x + cubeSize, y,            z + cubeSize, r, g, b, a};
                verts[v++] = VBOVertex{x,            y,            z + cubeSize, r, g, b, a};

                // Зад
                verts[v++] = VBOVertex{x,            y + cubeSize, z           , r, g, b, a};
                verts[v++] = VBOVertex{x,            y + cubeSize, z + cubeSize, r, g, b, a};
                verts[v++] = VBOVertex{x + cubeSize, y + cubeSize, z + cubeSize, r, g, b, a};
                verts[v++] = VBOVertex{x + cubeSize, y + cubeSize, z           , r, g, b, a};

                // Верх
                verts[v++] = VBOVertex{x,            y,            z + cubeSize, r, g, b, a};
                verts[v++] = VBOVertex{x + cubeSize, y,            z + cubeSize, r, g, b, a};
                verts[v++] = VBOVertex{x + cubeSize, y + cubeSize, z + cubeSize, r, g, b, a};
                verts[v++] = VBOVertex{x,            y + cubeSize, z + cubeSize, r, g, b, a};

                // Низ
                verts[v++] = VBOVertex{x,            y,            z           , r, g, b, a};
                verts[v++] = VBOVertex{x,            y + cubeSize, z           , r, g, b, a};
                verts[v++] = VBOVertex{x + cubeSize, y + cubeSize, z           , r, g, b, a};
                verts[v++] = VBOVertex{x + cubeSize, y,            z           , r, g, b, a};

                // Лево
                verts[v++] = VBOVertex{x,            y,            z           , r, g, b, a};
                verts[v++] = VBOVertex{x,            y,            z + cubeSize, r, g, b, a};
                verts[v++] = VBOVertex{x,            y + cubeSize, z + cubeSize, r, g, b, a};
                verts[v++] = VBOVertex{x,            y + cubeSize, z           , r, g, b, a};

                // Право
                verts[v++] = VBOVertex{x + cubeSize, y,            z           , r, g, b, a};
                verts[v++] = VBOVertex{x + cubeSize, y + cubeSize, z           , r, g, b, a};
                verts[v++] = VBOVertex{x + cubeSize, y + cubeSize, z + cubeSize, r, g, b, a};
                verts[v++] = VBOVertex{x + cubeSize, y,            z + cubeSize, r, g, b, a};
            }
        }
    }
}

__global__
void kernelUpdateVBO(VBOVertex *verts, uchar *data, uint xSize, uint ySize, uint zSize)
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
                uint index = getIndex(xSize, ySize, zSize, i, j, k);
                uchar r, g, b, a;

                switch (data[index])
                {
                    case 0: r = 0;   g = 255; b = 0;   a = 0; break;
                    case 1: r = 255; g = 255; b = 255; a = 0; break;
                    case 2: r = 255; g = 0;   b = 0;   a = 0; break;
                }

                for (uint v = index * 24; v < index * 24 + 24; v++)
                {
                    verts[v].r = r; verts[v].g = g; verts[v].b = b; verts[v].a = a;
                }
            }
        }
    }
}

void cudaInitGrid(Grid *g)
{
    dim3 blocks(
        ceil(g->xSize / float(threads.x)),
        ceil(g->ySize / float(threads.y)),
        ceil(g->zSize / float(threads.z))
    );

    uint x = blocks.x * threads.x;
    uint y = blocks.y * threads.y;
    uint z = blocks.z * threads.z;
    cudaMalloc((void **)& (g->randomStates), sizeof(curandState) * x * y * z);

    uint dataSize = g->xSize * g->ySize * g->zSize;
    cudaMalloc((void **)& (g->deviceMatrix), sizeof(uchar) * dataSize);
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
    uint dataSize = g->xSize * g->ySize * g->zSize;
    cudaMemcpy(g->hostMatrix, g->deviceMatrix, sizeof(uchar) * dataSize, cudaMemcpyDeviceToHost);
}

void cudaAlgorithmStep(Grid *g, uint algorithmSteps)
{
    dim3 blocks(
        ceil(g->xSize / float(threads.x)),
        ceil(g->ySize / float(threads.y)),
        ceil(g->zSize / float(threads.z))
    );

    for (uint i = 0; i < algorithmSteps; i++)
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

void cudaInitVBO(Grid *g, struct cudaGraphicsResource **cuda_resource, float percentOfCube)
{
    dim3 blocks(
        ceil(g->xSize / float(threads.x)),
        ceil(g->ySize / float(threads.y)),
        ceil(g->zSize / float(threads.z))
    );

    // Map buffer object
    VBOVertex *dev_ptr;
    size_t num_bytes;
    cudaGraphicsMapResources(1, cuda_resource, 0);
    cudaGraphicsResourceGetMappedPointer((void **)&dev_ptr, &num_bytes, *cuda_resource);

    kernelInitVBO<<<blocks, threads>>>(dev_ptr, g->deviceMatrix, g->xSize, g->ySize, g->zSize, percentOfCube);

    // Unmap buffer object
    cudaGraphicsUnmapResources(1, cuda_resource, 0);
}

void cudaUpdateVBO(Grid *g, struct cudaGraphicsResource **cuda_resource)
{
    dim3 blocks(
        ceil(g->xSize / float(threads.x)),
        ceil(g->ySize / float(threads.y)),
        ceil(g->zSize / float(threads.z))
    );

    // Map buffer object
    VBOVertex *dev_ptr;
    size_t num_bytes;
    cudaGraphicsMapResources(1, cuda_resource, 0);
    cudaGraphicsResourceGetMappedPointer((void **)&dev_ptr, &num_bytes, *cuda_resource);

    kernelUpdateVBO<<<blocks, threads>>>(dev_ptr, g->deviceMatrix, g->xSize, g->ySize, g->zSize);

    // Unmap buffer object
    cudaGraphicsUnmapResources(1, cuda_resource, 0);
}
