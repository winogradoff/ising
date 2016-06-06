#ifndef PLOTWIDGET_H
#define PLOTWIDGET_H

#include <QWidget>
#include "qcustomplot.h"
#include "grid.h"

#define PLOT_SIZE 100

namespace Ui
{
class PlotWidget;
}

class PlotWidget : public QWidget
{
    Q_OBJECT

public:
    PlotWidget();
    ~PlotWidget();

    void setGrid(Grid g);
    void setParams(Grid grid);
    void updatePlots(double energy, double magnetization);

protected:
    void closeEvent(QCloseEvent* event);

private:
    Ui::PlotWidget* ui;

    Grid grid;
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

    static void updateMinMax(double value, double& min, double& max);
    static void updateMinMax(double& min, double& max);

signals:
    void closedSignal();
};

#endif // PLOTWIDGET_H
