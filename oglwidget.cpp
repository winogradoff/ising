#include <QtGui>
#include <QtOpenGL>
#include <QMouseEvent>
#include "oglwidget.h"

// Параматры лампы
GLfloat OGLWidget::light_ambient[] = {0.0f, 0.0f, 0.0f};
GLfloat OGLWidget::light_diffuse[] = {1.0f, 1.0f, 1.0f};
GLfloat OGLWidget::light_specular[] = {1.0f, 1.0f, 1.0f};
GLfloat OGLWidget::light_position[] = {0.0f, 0.0f, 100.0f, 0.0f};
GLfloat OGLWidget::light_spot_direction[] = {0.0f, 0.0f, -1.0f, 1.0f};
GLfloat OGLWidget::light_spot_cutoff = 30.0;
GLfloat OGLWidget::light_spot_exponent = 15.0;

OGLWidget::OGLWidget(QWidget *parent) : QGLWidget(parent)
{
    this->xRot = 0.0;
    this->yRot = 0.0;
    this->zRot = 0.0;
    this->vertices = NULL;
}

OGLWidget::~OGLWidget()
{
    // TODO
    if (this->vertices != NULL)
    {
        delete[] this->vertices;
    }
}

void OGLWidget::initializeGL()
{
    initializeOpenGLFunctions();

    glClearColor(0, 0, 0, 1);

    // Тест грубины
    glEnable(GL_DEPTH_TEST);

    // Удаление невидимых граней
    glEnable(GL_CULL_FACE);

    //    // Приведение нормалей к единичной длине
    //    glEnable(GL_NORMALIZE);

    // Обход против часововй
    glFrontFace(GL_CCW);

    // Сглаживание
    glEnable(GL_MULTISAMPLE);

    // Закрашивание
    //    glShadeModel(GL_SMOOTH);
    //    glShadeModel(GL_FLAT);
    glEnable(GL_COLOR_MATERIAL);

    glLightModeli(GL_LIGHT_MODEL_TWO_SIDE, GL_FALSE);

    // Инициализация освещения
    glEnable(GL_LIGHTING);
    initializeLight();

    // Проверка VBO
    GLuint vbo;
    struct cudaGraphicsResource *cuda_vbo_resource;
    this->createVBO(&vbo, &cuda_vbo_resource, cudaGraphicsMapFlagsWriteDiscard);
    cudaTestVBO(&cuda_vbo_resource);
    this->deleteVBO(&vbo, cuda_vbo_resource);
}

void OGLWidget::paintGL()
{
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    glLoadIdentity();

    // Повернуть
    glRotatef(xRot, 1.0, 0.0, 0.0);
    glRotatef(yRot, 0.0, 1.0, 0.0);
    glRotatef(zRot, 0.0, 0.0, 1.0);

    glPushMatrix();

    glTranslatef(0.0, 0.0, 0.0);
    glScalef(1.0, 1.0, 1.0);
    glRotatef(0.0, 1.0, 0.0, 0.0);
    glRotatef(0.0, 0.0, 1.0, 0.0);
    glRotatef(0.0, 0.0, 0.0, 1.0);

    // Оси OX, OY, OZ
    this->drawAxes();

//    glColor4f(1.0, 1.0, 1.0, 1.0);
    this->drawFigure();

    glPopMatrix();
}

void OGLWidget::resizeGL(int width, int height)
{
    glViewport(0, 0, width, height);

    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();

    GLdouble aspect = (GLdouble) width / (GLdouble) height;
    gluPerspective(40.0, aspect, 1.0, 50.0);

    gluLookAt(0.0,  0.0, 10.0,  // координаты позиции глаза налюдателя
              0.0,  0.0, 0.0,  // координаты точки, распологающейся в центре экрана
              0.0,  1.0, 0.0); // направление вектора, задающего верх

    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();
}

// Инициализация освещения
void OGLWidget::initializeLight()
{
    glEnable(GL_LIGHT0);
    glLightfv(GL_LIGHT0, GL_AMBIENT,        OGLWidget::light_ambient);
    glLightfv(GL_LIGHT0, GL_DIFFUSE,        OGLWidget::light_diffuse);
    glLightfv(GL_LIGHT0, GL_SPECULAR,       OGLWidget::light_specular);
    glLightfv(GL_LIGHT0, GL_POSITION,       OGLWidget::light_position);
    glLightfv(GL_LIGHT0, GL_SPOT_DIRECTION, OGLWidget::light_spot_direction );
    glLightf(GL_LIGHT0,  GL_SPOT_CUTOFF,    OGLWidget::light_spot_cutoff);
    glLightf(GL_LIGHT0,  GL_SPOT_EXPONENT,  OGLWidget::light_spot_exponent);
}

// Рисование осей координат
void OGLWidget::drawAxes()
{
    // Отключить освещение для этого объекта
    glPushAttrib(GL_LIGHTING_BIT);
    glDisable(GL_LIGHTING);

    float axis_length = 3.0f;
    float nib_size = 0.05f;
    glLineWidth(3.0f);

    // OX
    glColor3f(1.0, 0.0, 0.0);
    glBegin(GL_LINES);
    glVertex3f(-axis_length, 0.0f, 0.0f);
    glVertex3f( axis_length, 0.0f, 0.0f);
    glEnd();
    glBegin(GL_LINE_STRIP);
    glVertex3f(axis_length - nib_size, -nib_size, 0.0f);
    glVertex3f(axis_length, 0.0f, 0.0f);
    glVertex3f(axis_length - nib_size, nib_size, 0.0f);
    glEnd();

    // OY
    glColor3f(0.0, 1.0, 0.0);
    glBegin(GL_LINES);
    glVertex3f(0.0, -axis_length, 0.0);
    glVertex3f(0.0, axis_length, 0.0);
    glEnd();
    glBegin(GL_LINE_STRIP);
    glVertex3f(-nib_size, axis_length - nib_size, 0.0f);
    glVertex3f(0.0, axis_length, 0.0);
    glVertex3f(nib_size, axis_length - nib_size, 0.0f);
    glEnd();

    // OZ
    glColor3f(0.0, 0.0, 1.0);
    glBegin(GL_LINES);
    glVertex3f(0.0, 0.0, -axis_length);
    glVertex3f(0.0, 0.0, axis_length);
    glEnd();
    glBegin(GL_LINE_STRIP);
    glVertex3f(-nib_size, -nib_size, axis_length - nib_size);
    glVertex3f(0.0f, 0.0f, axis_length);
    glVertex3f(nib_size, nib_size, axis_length - nib_size);
    glEnd();

    glPopAttrib();
}

// Рисование фигуры
void OGLWidget::drawFigure()
{
    // Отключить освещение для этого объекта
    glPushAttrib(GL_LIGHTING_BIT);
    glDisable(GL_LIGHTING);

    if (this->vertices != NULL)
    {
        GLfloat size = this->cubeSize;

        for(unsigned int i = 0; i <= this->verticesSize; i += 3)
        {
            GLfloat x = this->vertices[i    ] - 0.5f * size;
            GLfloat y = this->vertices[i + 1] - 0.5f * size;
            GLfloat z = this->vertices[i + 2] - 0.5f * size;

            switch(this->grid.hostMatrix[i / 3])
            {
                case 0: glColor3f(0.0f, 1.0f, 0.0f); break;
                case 2: glColor3f(1.0f, 0.0f, 0.0f); break;
                default: glColor3f(1.0f, 1.0f, 1.0f);
            }

            // Перед
            glBegin(GL_QUADS);
//                glNormal3f(0.0f, -1.0f, 0.0f);
                glVertex3f(x, y, z);
                glVertex3f(x + size, y, z);
                glVertex3f(x + size, y, z + size);
                glVertex3f(x, y, z + size);
            glEnd();

            // Зад
            glBegin(GL_QUADS);
//                glNormal3f(0.0f, 1.0f, 0.0f);
                glVertex3f(x, y + size, z);
                glVertex3f(x, y + size, z + size);
                glVertex3f(x + size, y + size, z + size);
                glVertex3f(x + size, y + size, z);
            glEnd();

            // Верх
            glBegin(GL_QUADS);
//                glNormal3f(0.0f, 0.0f, 1.0f);
                glVertex3f(x, y, z + size);
                glVertex3f(x + size, y, z + size);
                glVertex3f(x + size, y + size, z + size);
                glVertex3f(x, y + size, z + size);
            glEnd();

            // Низ
            glBegin(GL_QUADS);
//                glNormal3f(0.0f, 0.0f, -1.0f);
                glVertex3f(x, y, z);
                glVertex3f(x, y + size, z);
                glVertex3f(x + size, y + size, z);
                glVertex3f(x + size, y, z);
            glEnd();

            // Лево
            glBegin(GL_QUADS);
//                glNormal3f(-1.0f, 0.0f, 0.0f);
                glVertex3f(x, y, z);
                glVertex3f(x, y, z + size);
                glVertex3f(x, y + size, z + size);
                glVertex3f(x, y + size, z);
            glEnd();

            // Право
            glBegin(GL_QUADS);
//                glNormal3f(1.0f, 0.0f, 0.0f);
                glVertex3f(x + size, y, z);
                glVertex3f(x + size, y + size, z);
                glVertex3f(x + size, y + size, z + size);
                glVertex3f(x + size, y, z + size);
            glEnd();
        }
    }

    glPopAttrib();
}

void OGLWidget::setGrid(Grid grid)
{
    this->grid = grid;

    GLfloat percentOfCube = 0.3f;
    GLfloat gridSize = 5.0f;

    GLfloat countX = this->grid.xSize;
    GLfloat countY = this->grid.ySize;
    GLfloat countZ = this->grid.zSize;

    // TODO: максимальный count
    GLfloat cubeSize = (gridSize / countX) * percentOfCube;
    GLfloat cubeSpace = (gridSize / countX) * (1.0f - percentOfCube);

    GLfloat lengthX = countX * cubeSize + (countX - 1) * cubeSpace;
    GLfloat lengthY = countY * cubeSize + (countY - 1) * cubeSpace;
    GLfloat lengthZ = countZ * cubeSize + (countZ - 1) * cubeSpace;

    this->cubeSize = cubeSize;

    if (this->vertices != NULL)
    {
        delete[] this->vertices;
        this->vertices = NULL;
    }

    this->verticesSize = countX * countY * countZ * 3;
    this->vertices = new GLfloat[this->verticesSize];

    unsigned int pos = 0;
    for (int i = 0; i < countX; i++)
    {
        for (int j = 0; j < countY; j++)
        {
            for (int k = 0; k < countZ; k++)
            {
                this->vertices[pos++] = i * (cubeSize + cubeSpace) + 0.5f * (cubeSize - lengthX);
                this->vertices[pos++] = j * (cubeSize + cubeSpace) + 0.5f * (cubeSize - lengthY);
                this->vertices[pos++] = k * (cubeSize + cubeSpace) + 0.5f * (cubeSize - lengthZ);
            }
        }
    }

    this->update();
}

void OGLWidget::createVBO(GLuint *vbo, struct cudaGraphicsResource **vbo_res, unsigned int vbo_res_flags)
{
    // create buffer object
    glGenBuffers(1, vbo);
    glBindBuffer(GL_ARRAY_BUFFER, *vbo);

    // initialize buffer object
    unsigned int size = 100 * 100 * 4 * sizeof(float);
    glBufferData(GL_ARRAY_BUFFER, size, 0, GL_DYNAMIC_DRAW);

    glBindBuffer(GL_ARRAY_BUFFER, 0);

    // register this buffer object with CUDA
    cudaGraphicsGLRegisterBuffer(vbo_res, *vbo, vbo_res_flags);
}

void OGLWidget::deleteVBO(GLuint *vbo, struct cudaGraphicsResource *vbo_res)
{
    // unregister this buffer object with CUDA
    cudaGraphicsUnregisterResource(vbo_res);
    glBindBuffer(1, *vbo);
    glDeleteBuffers(1, vbo);
    *vbo = 0;
}

// Обработка движений мыши
static void qNormalizeAngle(GLfloat &angle)
{
    while (angle < 0)
    {
        angle += 360;
    }

    while (angle > 360)
    {
        angle -= 360;
    }
}

void OGLWidget::setXRotation(GLfloat angle)
{
    qNormalizeAngle(angle);
    if (angle != xRot)
    {
        xRot = angle;
        //        emit xRotationChanged(angle);
        update();
    }
}

void OGLWidget::setYRotation(GLfloat angle)
{
    qNormalizeAngle(angle);
    if (angle != yRot)
    {
        yRot = angle;
        //        emit yRotationChanged(angle);
        update();
    }
}

void OGLWidget::setZRotation(GLfloat angle)
{
    qNormalizeAngle(angle);
    if (angle != zRot)
    {
        zRot = angle;
        //        emit zRotationChanged(angle);
        update();
    }
}

void OGLWidget::mousePressEvent(QMouseEvent *event)
{
    lastPos = event->pos();
}

void OGLWidget::mouseMoveEvent(QMouseEvent *event)
{
    GLfloat dx = GLfloat( event->x() - lastPos.x() ) / 5.0;
    GLfloat dy = GLfloat( event->y() - lastPos.y() ) / 5.0;

    setXRotation(xRot + dy);

    if(event->buttons() & Qt::LeftButton)
    {
        setYRotation(yRot + dx);
    }
//    else if(event->buttons() & Qt::RightButton)
//    {
//        setZRotation(zRot + dx);
//    }

    lastPos = event->pos();
}
