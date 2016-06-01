#ifndef OGLWIDGET_H
#define OGLWIDGET_H

#include <QWidget>
#include <QOpenGLWidget>
#include <GL/glu.h>
#include "grid.h"

class OGLWidget : public QOpenGLWidget
{
public:
    OGLWidget(QWidget *parent = 0);
    ~OGLWidget();
    void setGrid(Grid grid);

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

private:
    GLfloat xRot;
    GLfloat yRot;
    GLfloat zRot;
    QPoint lastPos;

    Grid grid;
    GLfloat *vertices;
    unsigned int verticesSize;
    GLfloat cubeSize;

    // Инициализация освещения
    void initializeLight();

    // Рисование осей координат
    void drawAxes();

    // Рисование фигур
    void drawFigure();

    // Параметры ламп
    static GLfloat light_ambient[];
    static GLfloat light_specular[];
    static GLfloat light_diffuse[];
    static GLfloat light_position[];
    static GLfloat light_spot_direction[];
    static GLfloat light_spot_cutoff;
    static GLfloat light_spot_exponent;
};

#endif // OGLWIDGET_H
