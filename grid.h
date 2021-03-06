#ifndef GRID_H
#define GRID_H

#include <cuda.h>

// cuRAND
#include <curand.h>
#include <curand_kernel.h>

#include "types.h"

struct Grid
{
    DimensionEnum dimension; // размерность решётки
    int xSize, ySize, zSize; // размеры решётки
    int interactionEnergy; // энергия взаимодействия между спинами
    double externalField; // значение внешнего магнитного поля
    int interactionRadius; // радиус взаимодействия
    uint percentOfNonmagnetic; // количество немагнитных частиц
    double temperature; // температура

    curandState* randomStates; // состояния генераторов случайных чисел CUDA
    uchar* deviceMatrix; // решётка на GPU

    int percentOfCube; // размер для визуализации в %
};

// CUDA functions
void cudaInitGrid(Grid* g);
void cudaFreeGrid(Grid* g);
void cudaSetParams(Grid* g);
void cudaAlgorithmStep(Grid* g, uint algorithmSteps);
double cudaMagnetization(Grid* g);
double cudaEnergy(Grid* g);

void cudaInitVBO(Grid* g, struct cudaGraphicsResource** cudaVertexResource,
    struct cudaGraphicsResource** cudaIndexResource);

void cudaUpdateVBO(Grid* g, struct cudaGraphicsResource** cudaVertexResource);

#endif // GRID_H
