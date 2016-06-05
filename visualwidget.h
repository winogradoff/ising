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

    OGLWidget* OpenGLWGT;

protected:
    void closeEvent(QCloseEvent* event);

private:
    Ui::VisualWidget* ui;

signals:
    void closedSignal();
};

#endif // VISUALWIDGET_H
