QT += core gui opengl widgets printsupport concurrent

TARGET = curs_visual
TEMPLATE = app

HEADERS += widget.h \
    grid.h \
    qcustomplot.h \
    oglwidget.h

SOURCES += main.cpp\
    widget.cpp \
    qcustomplot.cpp \
    oglwidget.cpp

FORMS += widget.ui

win32 {
    RC_ICONS = icon.ico
}

# With C++11 support
CONFIG += c++11

DESTDIR = release
OBJECTS_DIR = release/obj
CUDA_OBJECTS_DIR = release/cuda

CUDA_SOURCES += cuda.cu

CUDA_DIR = "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v7.5"
SYSTEM_NAME = Win32
SYSTEM_TYPE = 32
CUDA_ARCH = compute_20
NVCC_OPTIONS = --use_fast_math

# include paths
INCLUDEPATH += $$CUDA_DIR/include

# library directories
QMAKE_LIBDIR += $$CUDA_DIR/lib/$$SYSTEM_NAME

# Add the necessary libraries
CUDA_LIBS = cuda cudart
CUDA_INC = $$join(INCLUDEPATH,'" -I"','-I"','"')
NVCC_LIBS = -lcuda -lcudart
LIBS += cuda.lib cudart.lib opengl32.lib glu32.lib

MSVCRT_LINK_FLAG_DEBUG = "/MTd"
MSVCRT_LINK_FLAG_RELEASE = "/MT"

# Configuration of the Cuda compiler
CONFIG(debug, debug|release) {
    # Debug mode
    cuda_d.input = CUDA_SOURCES
    cuda_d.output = $$CUDA_OBJECTS_DIR/${QMAKE_FILE_BASE}_cuda.obj
    cuda_d.commands = $$CUDA_DIR/bin/nvcc.exe \
                      -D_DEBUG $$NVCC_OPTIONS \
                      $$CUDA_INC $$NVCC_LIBS \
                      --machine $$SYSTEM_TYPE -arch=$$CUDA_ARCH \
                      -Xcompiler $$MSVCRT_LINK_FLAG_DEBUG \
                      -c -o ${QMAKE_FILE_OUT} ${QMAKE_FILE_NAME}
    cuda_d.dependency_type = TYPE_C
    QMAKE_EXTRA_COMPILERS += cuda_d
}
else {
    # Release mode
    cuda.input = CUDA_SOURCES
    cuda.output = $$CUDA_OBJECTS_DIR/${QMAKE_FILE_BASE}_cuda.obj
    cuda.commands = $$CUDA_DIR/bin/nvcc.exe \
                    $$NVCC_OPTIONS \
                    $$CUDA_INC $$NVCC_LIBS \
                    --machine $$SYSTEM_TYPE -arch=$$CUDA_ARCH \
                    -Xcompiler $$MSVCRT_LINK_FLAG_RELEASE \
                    -c -o ${QMAKE_FILE_OUT} ${QMAKE_FILE_NAME}
    cuda.dependency_type = TYPE_C
    QMAKE_EXTRA_COMPILERS += cuda
}
