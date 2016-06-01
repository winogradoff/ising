#ifndef GRID_H
#define GRID_H

#include <cuda.h>

// cuRAND
#include <curand.h>
#include <curand_kernel.h>

typedef unsigned char BYTE;

enum DimensionEnum {DIM_1, DIM_2, DIM_3};

struct Grid
{
    DimensionEnum dimension; // размерность решётки
    int xSize, ySize, zSize; // размеры решётки
    int interactionEnergy; // энергия взаимодействия между спинами
    double externalField; // значение внешнего магнитного поля
    int interactionRadius; // радиус взаимодействия
    long nonmagneticParticles; // количество немагнитных частиц
    double temperature; // температура

    curandState *randomStates; // состояния генераторов случайных чисел CUDA
    BYTE *hostMatrix; // решётка на CPU
    BYTE *deviceMatrix; // решётка на GPU
};

#endif // GRID_H
