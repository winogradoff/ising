#ifndef TYPES
#define TYPES

#define BUFFER_OFFSET(i) ((void*)(i))

typedef unsigned char uchar;
typedef unsigned short ushort;
typedef unsigned int uint;

enum DimensionEnum
{
    DIM_1,
    DIM_2,
    DIM_3
};

struct VBOVertex
{
    float x, y, z; // координаты
    uchar r, g, b, a; // цвет
};

struct ViewerPosition
{
    float x, y, z;
};

#endif // TYPES
