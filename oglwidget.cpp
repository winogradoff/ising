#include "oglwidget.h"

OGLWidget::OGLWidget(QWidget *parent) : QGLWidget(parent)
{
    this->glInitialized = false;
    this->xRot = 0.0;
    this->yRot = 0.0;
    this->zRot = 0.0;
    this->viewerPosition = ViewerPosition{0.0, 0.0, 10.0};

    this->VertexVBOID = 0;
    this->IndexVBOID = 0;
    this->cudaVertexResource = NULL;
    this->cudaIndexResource = NULL;
}

OGLWidget::~OGLWidget()
{
    this->deleteVBO();
}

void OGLWidget::createVBO()
{
    uint size = this->grid.xSize * this->grid.ySize * this->grid.zSize;

    // 8 реальных вершин на каждый куб
    uint vertexSize = size * sizeof(VBOVertex) * 8;

    // 24 индекса вершин на каждый куб для рисования GL_QUADS
    uint indexSize = size * sizeof(uint) * 24;

    // Создать VBO для вершин
    glGenBuffers(1, &(this->VertexVBOID));
    glBindBuffer(GL_ARRAY_BUFFER, this->VertexVBOID);
    glBufferData(GL_ARRAY_BUFFER, vertexSize, 0, GL_DYNAMIC_DRAW); // TODO
    glBindBuffer(GL_ARRAY_BUFFER, 0);

    // Создать VBO для индексов
    glGenBuffers(1, &(this->IndexVBOID));
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, this->IndexVBOID);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, indexSize, 0, GL_DYNAMIC_DRAW); // TODO
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);

    // Подключить вершины и индексы к CUDA
    cudaGraphicsGLRegisterBuffer(
        &(this->cudaVertexResource),
        this->VertexVBOID,
        cudaGraphicsMapFlagsWriteDiscard
    );
    cudaGraphicsGLRegisterBuffer(
        &(this->cudaIndexResource),
        this->IndexVBOID,
        cudaGraphicsMapFlagsWriteDiscard
    );
}

void OGLWidget::updateVBO()
{
    if (this->VertexVBOID)
    {
        cudaUpdateVBO(&(this->grid), &(this->cudaVertexResource));
    }
}

void OGLWidget::deleteVBO()
{
    if (this->cudaVertexResource != NULL)
    {
        cudaGraphicsUnregisterResource(this->cudaVertexResource);
        this->cudaVertexResource = NULL;
    }

    if (this->cudaIndexResource != NULL)
    {
        cudaGraphicsUnregisterResource(this->cudaIndexResource);
        this->cudaIndexResource = NULL;
    }

    if (this->VertexVBOID)
    {
        glBindBuffer(1, this->VertexVBOID);
        glDeleteBuffers(1, &(this->VertexVBOID));
        this->VertexVBOID = 0;
    }

    if (this->IndexVBOID)
    {
        glBindBuffer(1, this->IndexVBOID);
        glDeleteBuffers(1, &(this->IndexVBOID));
        this->IndexVBOID = 0;
    }
}

void OGLWidget::initializeGL()
{
    initializeOpenGLFunctions();
    this->glInitialized = true;

    glClearColor(0, 0, 0, 1);

    // Тест грубины
    glEnable(GL_DEPTH_TEST);

    // Удаление невидимых граней
    glEnable(GL_CULL_FACE);

    // Обход против часововй
    glFrontFace(GL_CCW);

    // Сглаживание
    glEnable(GL_MULTISAMPLE);

//    // Закрашивание
//    glShadeModel(GL_SMOOTH);
//    glShadeModel(GL_FLAT);
//    glEnable(GL_COLOR_MATERIAL);
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

        glPushAttrib(GL_LIGHTING_BIT);
        glDisable(GL_LIGHTING);

        this->drawAxes();
        this->drawFigure();

        glPopAttrib();

    glPopMatrix();
}

void OGLWidget::resizeGL(int width, int height)
{
    glViewport(0, 0, width, height);

    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();

    GLdouble aspect = (GLdouble) width / (GLdouble) height;
    gluPerspective(45.0, aspect, 0.01, 100.0);

    gluLookAt(
        // координаты позиции глаза налюдателя
        this->viewerPosition.x,  this->viewerPosition.y, this->viewerPosition.z,
        // координаты точки, распологающейся в центре экрана
        this->viewerPosition.x,  this->viewerPosition.y, 0.0,
        // направление вектора, задающего верх
        0.0,  1.0, 0.0
    );

    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();
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
    glColor3f(0.7f, 0.0f, 0.0f);
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
    glColor3f(0.0f, 0.7f, 0.0f);
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
    glColor3f(0.0f, 0.0f, 0.7f);
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

void OGLWidget::drawFigure()
{
    if (this->VertexVBOID == 0 && this->glInitialized)
    {
        this->deleteVBO();
        this->createVBO();
        cudaInitVBO(
            &(this->grid),
            &(this->cudaVertexResource),
            &(this->cudaIndexResource),
            this->percentOfCube
        );
        this->update();
    }

//    glBindBuffer(GL_ARRAY_BUFFER, this->VertexVBOID);
//    glEnableClientState(GL_VERTEX_ARRAY);
//    glEnableClientState(GL_COLOR_ARRAY);

//    glVertexPointer(3, GL_FLOAT, 16, (void *) 0);
//    glColorPointer(4, GL_UNSIGNED_BYTE, 16, (void *) 12);

//    glDrawArrays(GL_QUADS, 0, this->grid.xSize * this->grid.ySize * this->grid.zSize * 24);

    glEnableClientState(GL_VERTEX_ARRAY);
    glEnableClientState(GL_COLOR_ARRAY);

    glBindBuffer(GL_ARRAY_BUFFER, this->VertexVBOID);

    glVertexPointer(3, GL_FLOAT, sizeof(VBOVertex), BUFFER_OFFSET(0));
    glColorPointer(4, GL_UNSIGNED_BYTE, sizeof(VBOVertex), BUFFER_OFFSET(12));

    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, this->IndexVBOID);

    glDrawElements(
        GL_QUADS,
        this->grid.xSize * this->grid.ySize * this->grid.zSize * 24,
        GL_UNSIGNED_INT,
        BUFFER_OFFSET(0)
    );

    glDisableClientState(GL_VERTEX_ARRAY);
    glDisableClientState(GL_COLOR_ARRAY);
}

void OGLWidget::setGrid(Grid grid)
{
    this->grid = grid;

    if (this->glInitialized)
    {
        this->deleteVBO();
        this->createVBO();
        cudaInitVBO(
            &(this->grid),
            &(this->cudaVertexResource),
            &(this->cudaIndexResource),
            this->percentOfCube
        );
    }
}

void OGLWidget::setCubeSize(int value)
{
    this->percentOfCube = value;

    if (this->VertexVBOID)
    {
        cudaInitVBO(
            &(this->grid),
            &(this->cudaVertexResource),
            &(this->cudaIndexResource),
            this->percentOfCube
        );
    }
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
        update();
    }
}

void OGLWidget::setYRotation(GLfloat angle)
{
    qNormalizeAngle(angle);
    if (angle != yRot)
    {
        yRot = angle;
        update();
    }
}

void OGLWidget::setZRotation(GLfloat angle)
{
    qNormalizeAngle(angle);
    if (angle != zRot)
    {
        zRot = angle;
        update();
    }
}

void OGLWidget::mousePressEvent(QMouseEvent *event)
{
    lastPos = event->pos();
}

void OGLWidget::mouseMoveEvent(QMouseEvent *event)
{
    GLfloat dx = GLfloat( event->x() - lastPos.x() ) / 10.0f;
    GLfloat dy = GLfloat( event->y() - lastPos.y() ) / 10.0f;

    setXRotation(xRot + dy);

    if(event->buttons() & Qt::LeftButton)
    {
        setYRotation(yRot + dx);
    }
    else if(event->buttons() & Qt::RightButton)
    {
        setZRotation(zRot + dx);
    }

    lastPos = event->pos();
}

void OGLWidget::wheelEvent(QWheelEvent *event)
{
    int delta = event->delta();

    if (delta < 0)
    {
        this->viewerPosition.z += 0.05f;
    }
    else if (delta > 0)
    {
        this->viewerPosition.z -= 0.05f;
    }

    if (this->viewerPosition.z < 0.02f) this->viewerPosition.z = 0.02f;
    if (this->viewerPosition.z > 20.0f) this->viewerPosition.z = 20.0f;

    this->resizeGL(this->width(), this->height());
    this->update();
}

void OGLWidget::keyPressEvent(QKeyEvent *event)
{
    switch(event->key())
    {
        case Qt::Key_Up:
        case Qt::Key_W:
            this->viewerPosition.y += 0.05f;
        break;
        case Qt::Key_Down:
        case Qt::Key_S:
            this->viewerPosition.y -= 0.05f;
        break;
        case Qt::Key_Left:
        case Qt::Key_A:
            this->viewerPosition.x -= 0.05f;
        break;
        case Qt::Key_Right:
        case Qt::Key_D:
            this->viewerPosition.x += 0.05f;
        break;
    }

    this->resizeGL(this->width(), this->height());
    this->update();
}
