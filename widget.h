#ifndef WIDGET_H
#define WIDGET_H

#include <QWidget>
#include <QtConcurrent/QtConcurrent>
#include <QGraphicsScene>
#include "grid.h"
#include "visualwidget.h"
#include "plotwidget.h"

class Widget;

namespace Ui
{
class Widget;
}

class Widget : public QWidget
{
    Q_OBJECT

public:
    explicit Widget(QWidget* parent = 0);
    ~Widget();

private:
    VisualWidget* VisualWGT;
    PlotWidget* PlotWGT;

    Grid grid;

    Ui::Widget* ui;

    int algorithmSteps;
    long long iterationNumber;

    QFutureWatcher<void> watcher;
    int state; // состояние системы (0 - стоп, 1 - старт)

    void newGrid();
    void run();
    void updateForm();
    void updateVisual();
    void updatePlots();

protected:
    void closeEvent(QCloseEvent* event);

public slots:
    void on_newButton_clicked();
    void on_updateButton_clicked();
    void on_startStopButton_clicked();
    void on_dimensions_currentIndexChanged(int index);
    void on_cubeSize_valueChanged(int value);
    void check();

private slots:
    void on_visualButton_clicked();
    void on_plotButton_clicked();
};

#endif // WIDGET_H
