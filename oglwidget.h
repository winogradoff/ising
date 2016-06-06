#ifndef OGLWIDGET_H
#define OGLWIDGET_H

#include <QtGui>
#include <QWidget>
#include <QGLWidget>
#include <QMouseEvent>
#include <QWheelEvent>
#include <QKeyEvent>
#include <QtOpenGL>
#include <QOpenGLFunctions>
#include <GL/glu.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_gl_interop.h>
#include "types.h"
#include "grid.h"

class OGLWidget : public QGLWidget, protected QOpenGLFunctions
{
public:
    OGLWidget(QWidget* parent = 0);
    ~OGLWidget();

    void setGrid(Grid g);
    void setCubeSize(int value);
    void setParams(Grid grid);

    // Vertex Buffer Object <-> CUDA
    GLuint VertexVBOID;
    GLuint IndexVBOID;
    struct cudaGraphicsResource* cudaVertexResource;
    struct cudaGraphicsResource* cudaIndexResource;
    void createVBO();
    void deleteVBO();
    void updateVBO();

public slots:
    void setXRotation(GLfloat angle);
    void setYRotation(GLfloat angle);
    void setZRotation(GLfloat angle);

signals:
    void xRotationChanged(GLfloat angle);
    void yRotationChanged(GLfloat angle);
    void zRotationChanged(GLfloat angle);

protected:
    void initializeGL();
    void resizeGL(int w, int h);
    void paintGL();
    void mousePressEvent(QMouseEvent*);
    void mouseMoveEvent(QMouseEvent*);
    void wheelEvent(QWheelEvent* event);
    void keyPressEvent(QKeyEvent* event);

private:
    bool glInitialized;

    GLfloat xRot;
    GLfloat yRot;
    GLfloat zRot;
    QPoint lastPos;

    Grid grid;
    int percentOfCube;

    ViewerPosition viewerPosition;

    // Рисование осей координат
    void drawAxes();

    // Рисование фигур
    void drawFigure();
};

#endif // OGLWIDGET_H
