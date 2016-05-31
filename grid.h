#ifndef GRID_H
#define GRID_H

#include <cuda.h>

// cuRAND
#include <curand.h>
#include <curand_kernel.h>

typedef unsigned char BYTE;

enum DimensionEnum {
    DIM_1,
    DIM_2,
    DIM_3
};

struct Grid
{
    DimensionEnum dimension;
    int xSize;
    int ySize;
    int zSize;
    int interactionEnergy; // энергия взаимодействия между спинами
    double externalField; // значение внешнего магнитного поля
    int interactionRadius;
    long nonmagneticParticles;
    double temperature;

    curandState *randomStates;
    BYTE *tempMatrix;
    BYTE *prevMatrix;
    BYTE *currMatrix;
};

#endif // GRID_H
