#include "plotwidget.h"
#include "ui_plotwidget.h"

PlotWidget::PlotWidget()
    : ui(new Ui::PlotWidget)
{
    ui->setupUi(this);
}

PlotWidget::~PlotWidget()
{
    delete ui;
}

void PlotWidget::setGrid(Grid g)
{
    this->grid = g;

    this->energy = 0.0;
    this->energySquare = 0.0;
    this->magnetization = 0.0;
    this->magnetizationSquare = 0.0;

    this->counter = 0;
    this->realStepList.clear();
    this->realEnergyList.clear();
    this->realMagnetizationList.clear();

    this->chartEnergyList.clear();
    this->chartMagnetizationList.clear();
    this->chartHeatCapacityList.clear();
    this->chartMagneticSusceptibilityList.clear();

    ui->customPlot1->clearGraphs();
    ui->customPlot2->clearGraphs();
    ui->customPlot3->clearGraphs();
    ui->customPlot4->clearGraphs();

    ui->customPlot1->replot();
    ui->customPlot2->replot();
    ui->customPlot3->replot();
    ui->customPlot4->replot();
}

void PlotWidget::setParams(Grid g)
{
    this->grid.externalField = g.externalField;
    this->grid.interactionEnergy = g.interactionEnergy;
    this->grid.interactionRadius = g.interactionRadius;
    this->grid.temperature = g.temperature;
}

void PlotWidget::updatePlots(double e, double m)
{
    double temperature = this->grid.temperature;

    this->realStepList << ++this->counter;
    this->realEnergyList << e;
    this->realMagnetizationList << m;

    this->energy += e;
    this->energySquare += e * e;
    this->magnetization += m;
    this->magnetizationSquare += m * m;

    int listSize = this->realStepList.size();
    int calcSize = qMin(PLOT_SIZE, listSize);

    if (listSize > calcSize)
    {
        this->realStepList.takeFirst();
        this->chartEnergyList.takeFirst();
        this->chartMagnetizationList.takeFirst();
        this->chartHeatCapacityList.takeFirst();
        this->chartMagneticSusceptibilityList.takeFirst();
        e = this->realEnergyList.takeFirst();
        m = this->realMagnetizationList.takeFirst();
        this->energy -= e;
        this->energySquare -= e * e;
        this->magnetization -= m;
        this->magnetizationSquare -= m * m;
    }

    double energy = this->energy / calcSize;
    double energySquare = this->energySquare / calcSize;
    double magnetization = this->magnetization / calcSize;
    double magnetizationSquare = this->magnetizationSquare / calcSize;

    // Расчёт теплоёмкости и магнитной восприимчивости
    double heatCapacity = (energySquare - energy * energy) / (temperature * temperature);
    double magneticSusceptibility
        = (magnetizationSquare - magnetization * magnetization) / temperature;

    // Средние значения на один спин (удельные)
    double gridTotal = this->grid.xSize * this->grid.ySize * this->grid.zSize;
    this->chartEnergyList << energy / gridTotal;
    this->chartMagnetizationList << magnetization / gridTotal;
    this->chartHeatCapacityList << heatCapacity / gridTotal;
    this->chartMagneticSusceptibilityList << magneticSusceptibility / gridTotal;

    QVector<double> stepVector(this->realStepList.toVector());
    QVector<double> energyVector(this->chartEnergyList.toVector());
    QVector<double> magnetizationVector(this->chartMagnetizationList.toVector());
    QVector<double> heatCapacityVector(this->chartHeatCapacityList.toVector());
    QVector<double> magneticSusceptibilityVector(this->chartMagneticSusceptibilityList.toVector());

    double stepMin, stepMax;
    double energyMin, energyMax;
    double magnetizationMin, magnetizationMax;
    double heatCapacityMin, heatCapacityMax;
    double magneticSusceptibilityMin, magneticSusceptibilityMax;
    stepMin = stepMax = 0.0;
    energyMin = energyMax = 0.0;
    magnetizationMin = magnetizationMax = 0.0;
    heatCapacityMin = heatCapacityMax = 0.0;
    magneticSusceptibilityMin = magneticSusceptibilityMax = 0.0;

    bool firstIteration = true;

    for (int i = 0; i < stepVector.size(); i++)
    {
        double step = stepVector[i];
        double energy = energyVector[i];
        double magnetization = magnetizationVector[i];
        double heatCapacity = heatCapacityVector[i];
        double magneticSusceptibility = magneticSusceptibilityVector[i];

        if (firstIteration)
        {
            stepMin = stepMax = step;
            energyMin = energyMax = energy;
            magnetizationMin = magnetizationMax = magnetization;
            heatCapacityMin = heatCapacityMax = heatCapacity;
            magneticSusceptibilityMin = magneticSusceptibilityMax = magneticSusceptibility;
            firstIteration = false;
        }
        else
        {
            this->updateMinMax(step, stepMin, stepMax);
            this->updateMinMax(energy, energyMin, energyMax);
            this->updateMinMax(magnetization, magnetizationMin, magnetizationMax);
            this->updateMinMax(heatCapacity, heatCapacityMin, heatCapacityMax);
            this->updateMinMax(
                magneticSusceptibility, magneticSusceptibilityMin, magneticSusceptibilityMax);
        }
    }

    this->updateMinMax(stepMin, stepMax);
    this->updateMinMax(energyMin, energyMax);
    this->updateMinMax(magnetizationMin, magnetizationMax);
    this->updateMinMax(heatCapacityMin, heatCapacityMax);
    this->updateMinMax(magneticSusceptibilityMin, magneticSusceptibilityMax);

    // График удельной энергии
    ui->customPlot1->clearGraphs();
    ui->customPlot1->addGraph();
    ui->customPlot1->graph(0)->setData(stepVector, energyVector);
    ui->customPlot1->graph(0)->setScatterStyle(
        QCPScatterStyle(QCPScatterStyle::ssNone, Qt::darkBlue, 5.0));
    ui->customPlot1->xAxis->setLabel("Шаг");
    ui->customPlot1->yAxis->setLabel("Энергия");
    ui->customPlot1->xAxis->setRange(stepMin, stepMax);
    ui->customPlot1->yAxis->setRange(energyMin, energyMax);
    ui->customPlot1->replot();

    // График удельной намагниченности
    ui->customPlot2->clearGraphs();
    ui->customPlot2->addGraph();
    ui->customPlot2->graph(0)->setData(stepVector, magnetizationVector);
    ui->customPlot2->graph(0)->setScatterStyle(
        QCPScatterStyle(QCPScatterStyle::ssNone, Qt::darkBlue, 5.0));
    ui->customPlot2->xAxis->setLabel("Шаг");
    ui->customPlot2->yAxis->setLabel("Намагниченность");
    ui->customPlot2->xAxis->setRange(stepMin, stepMax);
    ui->customPlot2->yAxis->setRange(magnetizationMin, magnetizationMax);
    ui->customPlot2->replot();

    // График удельной теплоёмкости
    ui->customPlot3->clearGraphs();
    ui->customPlot3->addGraph();
    ui->customPlot3->graph(0)->setData(stepVector, heatCapacityVector);
    ui->customPlot3->graph(0)->setScatterStyle(
        QCPScatterStyle(QCPScatterStyle::ssNone, Qt::darkBlue, 5.0));
    ui->customPlot3->xAxis->setLabel("Шаг");
    ui->customPlot3->yAxis->setLabel("Теплоёмкость");
    ui->customPlot3->xAxis->setRange(stepMin, stepMax);
    ui->customPlot3->yAxis->setRange(heatCapacityMin, heatCapacityMax);
    ui->customPlot3->replot();

    // График магнитной восприимчивости
    ui->customPlot4->clearGraphs();
    ui->customPlot4->addGraph();
    ui->customPlot4->graph(0)->setData(stepVector, magneticSusceptibilityVector);
    ui->customPlot4->graph(0)->setScatterStyle(
        QCPScatterStyle(QCPScatterStyle::ssNone, Qt::darkBlue, 5.0));
    ui->customPlot4->xAxis->setLabel("Шаг");
    ui->customPlot4->yAxis->setLabel("Восприимчивость");
    ui->customPlot4->xAxis->setRange(stepMin, stepMax);
    ui->customPlot4->yAxis->setRange(magneticSusceptibilityMin, magneticSusceptibilityMax);
    ui->customPlot4->replot();
}

void PlotWidget::closeEvent(QCloseEvent* e)
{
    QWidget::closeEvent(e);
    emit closedSignal();
}

void PlotWidget::updateMinMax(double value, double& min, double& max)
{
    if (value < min)
        min = value;
    if (value > max)
        max = value;
}

void PlotWidget::updateMinMax(double& min, double& max)
{
    if (qAbs(max - min) < std::numeric_limits<double>::epsilon())
    {
        min -= 0.001;
        max += 0.001;
    }
}
