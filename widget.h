#ifndef WIDGET_H
#define WIDGET_H

#include <QWidget>
#include <QtConcurrent/QtConcurrent>
#include <QGraphicsScene>
#include "grid.h"

#define PLOT_SIZE 100
//

class Widget;

namespace Ui {
    class Widget;
}

class Widget : public QWidget
{
    Q_OBJECT

public:
    explicit Widget(QWidget *parent = 0);
    ~Widget();

private:
    Grid grid;

    Ui::Widget *ui;
    QGraphicsScene* scene;

    int colors[3][3];
    int coordZ;

    int algorithmSteps;

    QFutureWatcher<void> watcher;
    int state; // состояние системы (0 - стоп, 1 - старт)

    int counter;
    QList<double> realStepList;
    QList<double> realEnergyList;
    QList<double> realMagnetizationList;

    double energy;
    double energySquare;
    double magnetization;
    double magnetizationSquare;

    QList<double> chartEnergyList;
    QList<double> chartMagnetizationList;
    QList<double> chartHeatCapacityList;
    QList<double> chartMagneticSusceptibilityList;

    void resizeEvent(QResizeEvent* event);
    void newGrid();
    void run();
    void updateForm();
    void updateImage();
    void updatePlots();
    static void updateMinMax(double value, double& min, double& max);
    static void updateMinMax(double& min, double& max);

public slots:
    void on_newButton_clicked();
    void on_startStopButton_clicked();
    void on_dimensions_currentIndexChanged(int index);
    void on_coordZ_valueChanged(int value);
    void on_cubeSize_valueChanged(int value);

    void check();
};

#endif // WIDGET_H
