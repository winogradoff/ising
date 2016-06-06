#ifndef VISUALWIDGET_H
#define VISUALWIDGET_H

#include <QWidget>
#include "oglwidget.h"

namespace Ui
{
class VisualWidget;
}

class VisualWidget : public QWidget
{
    Q_OBJECT

public:
    VisualWidget(QWidget* parent = 0);
    ~VisualWidget();

    void setGrid(Grid g);
    void setCubeSize(int value);
    void setParams(Grid g);
    void updateImage();

protected:
    void closeEvent(QCloseEvent* event);

private:
    Ui::VisualWidget* ui;

    OGLWidget* OpenGLWGT;

signals:
    void closedSignal();
};

#endif // VISUALWIDGET_H
