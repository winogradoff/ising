#ifndef TYPES
#define TYPES

typedef unsigned char uchar;
typedef unsigned int uint;

enum DimensionEnum {DIM_1, DIM_2, DIM_3};

struct VBOVertex
{
  float x, y, z;    // координаты
  uchar r, g, b, a; // цвет
};

struct ViewerPosition
{
    float x, y, z;
};

#endif // TYPES
