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
    OGLWidget(QWidget *parent = 0);
    ~OGLWidget();
    void setGrid(Grid grid);

    void setCubeSize(int value);

    // Vertex Buffer Object <-> CUDA
    GLuint vbo;
    struct cudaGraphicsResource *cuda_resource;
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
    void mousePressEvent( QMouseEvent * );
    void mouseMoveEvent( QMouseEvent * );
    void wheelEvent(QWheelEvent *event);
    void keyPressEvent(QKeyEvent *event);

private:
    bool glInitialized;

    GLfloat xRot;
    GLfloat yRot;
    GLfloat zRot;
    QPoint lastPos;

    Grid grid;
    GLfloat *vertices;
    uint verticesSize;

    GLfloat percentOfCube;

    ViewerPosition viewerPosition;

    // Параметры ламп
    static GLfloat light_ambient[];
    static GLfloat light_specular[];
    static GLfloat light_diffuse[];
    static GLfloat light_position[];
    static GLfloat light_spot_direction[];
    static GLfloat light_spot_cutoff;
    static GLfloat light_spot_exponent;

    // Инициализация освещения
    void initializeLight();

    // Рисование осей координат
    void drawAxes();

    // Рисование фигур
    void drawFigure();
};

#endif // OGLWIDGET_H
