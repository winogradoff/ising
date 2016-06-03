#include "widget.h"
#include "ui_widget.h"
#include <QMessageBox>
#include <QGraphicsPixmapItem>
#include <QGraphicsItem>
#include <QTime>
#include <QDebug>
#include <QImage>

Widget::Widget(QWidget *parent) : QWidget(parent), ui(new Ui::Widget)
{
    ui->setupUi(this);
    connect(&(this->watcher), &QFutureWatcher<void>::finished, this, &Widget::check);

    this->grid.randomStates = NULL;
    this->grid.hostMatrix = NULL;
    this->grid.deviceMatrix = NULL;

    this->state = 0;
    this->scene = NULL;
    this->coordZ = 0;

    this->colors[0][0] = 0;
    this->colors[0][1] = 128;
    this->colors[0][2] = 0;
    this->colors[1][0] = 255;
    this->colors[1][1] = 255;
    this->colors[1][2] = 255;
    this->colors[2][0] = 128;
    this->colors[2][1] = 0;
    this->colors[2][2] = 0;

    ui->coordZLabel->setVisible(false);
    ui->coordZ->setVisible(false);
    ui->graphicsView->setVisible(false);
    ui->customPlot1->setVisible(false);
    ui->customPlot2->setVisible(false);
    ui->customPlot3->setVisible(false);
    ui->customPlot4->setVisible(false);

    this->updateForm();
    this->on_newButton_clicked();
}

Widget::~Widget()
{
    delete ui;
    this->watcher.waitForFinished();
}

void Widget::on_dimensions_currentIndexChanged(int)
{
    this->updateForm();
}

void Widget::on_coordZ_valueChanged(int)
{
    this->updateForm();
    this->updateImage();
}

void Widget::on_cubeSize_valueChanged(int value)
{
    ui->openGLWidget->setCubeSize(value);
    ui->openGLWidget->update();
}

void Widget::on_newButton_clicked()
{
    this->newGrid();
    this->updateImage();

//    this->energy = 0.0;
//    this->energySquare = 0.0;
//    this->magnetization = 0.0;
//    this->magnetizationSquare = 0.0;

//    this->counter = 0;
//    this->realStepList.clear();
//    this->realEnergyList.clear();
//    this->realMagnetizationList.clear();

//    this->chartEnergyList.clear();
//    this->chartMagnetizationList.clear();
//    this->chartHeatCapacityList.clear();
//    this->chartMagneticSusceptibilityList.clear();

//    ui->customPlot1->clearGraphs();
//    ui->customPlot2->clearGraphs();
//    ui->customPlot3->clearGraphs();
//    ui->customPlot4->clearGraphs();

//    ui->customPlot1->replot();
//    ui->customPlot2->replot();
//    ui->customPlot3->replot();
//    ui->customPlot4->replot();
}

void Widget::on_startStopButton_clicked()
{
    if(this->state == 0)
    {
        this->state = 1;
        ui->newButton->setEnabled(false);
        ui->startStopButton->setText("Стоп");
        this->algorithmSteps = ui->algorithmSteps->value();
        this->run();
    }
    else
    {
        this->state = 0;
        this->watcher.waitForFinished();
        ui->newButton->setEnabled(true);
        ui->startStopButton->setText("Старт");
    }
}

void Widget::updateForm()
{
    switch (static_cast<DimensionEnum>(ui->dimensions->currentIndex()))
    {
        case DIM_1:
            ui->gridXLabel->setVisible(true);
            ui->gridXSize->setVisible(true);
            ui->gridYLabel->setVisible(false);
            ui->gridYSize->setVisible(false);
            ui->gridZLabel->setVisible(false);
            ui->gridZSize->setVisible(false);
//            ui->coordZLabel->setVisible(false);
//            ui->coordZ->setVisible(false);
        break;

        case DIM_2:
            ui->gridXLabel->setVisible(true);
            ui->gridXSize->setVisible(true);
            ui->gridYLabel->setVisible(true);
            ui->gridYSize->setVisible(true);
            ui->gridZLabel->setVisible(false);
            ui->gridZSize->setVisible(false);
//            ui->coordZLabel->setVisible(false);
//            ui->coordZ->setVisible(false);
        break;

        case DIM_3:
            ui->gridXLabel->setVisible(true);
            ui->gridXSize->setVisible(true);
            ui->gridYLabel->setVisible(true);
            ui->gridYSize->setVisible(true);
            ui->gridZLabel->setVisible(true);
            ui->gridZSize->setVisible(true);
//            ui->coordZLabel->setVisible(true);
//            ui->coordZ->setVisible(true);
        break;
    }

//    int newCoordZ = ui->coordZ->value();

//    if (newCoordZ != this->coordZ)
//    {
//        if (newCoordZ >= this->grid.zSize) {
//            newCoordZ = this->grid.zSize - 1;
//            ui->coordZ->setValue(newCoordZ);
//        }
//        this->coordZ = newCoordZ;
//    }
}

void Widget::newGrid()
{
    this->grid.dimension = static_cast<DimensionEnum>(ui->dimensions->currentIndex());
    this->grid.xSize = ui->gridXSize->value();
    this->grid.ySize = ui->gridYSize->value();
    this->grid.zSize = ui->gridZSize->value();
    this->grid.interactionEnergy = ui->interactionEnergy->value();
    this->grid.externalField = ui->externalField->value();
    this->grid.temperature = ui->temperature->value();
    this->grid.interactionRadius = ui->interactionRadius->value();
    this->grid.nonmagneticParticles = ui->nonmagneticParticles->value();
    this->algorithmSteps = ui->algorithmSteps->value();

    this->coordZ = ui->coordZ->value();

    switch (this->grid.dimension)
    {
        case DIM_1:
            this->grid.ySize = this->grid.zSize = 1;
            this->coordZ = 0;
        break;

        case DIM_2:
            this->grid.zSize = 1;
            this->coordZ = 0;
        break;

        case DIM_3:
        break;
    }

    cudaFreeGrid(&(this->grid));
    if (this->grid.hostMatrix != NULL) delete[] this->grid.hostMatrix;

    cudaInitGrid(&(this->grid));
    this->grid.hostMatrix = new BYTE[this->grid.xSize * this->grid.ySize * this->grid.zSize];

    ui->openGLWidget->setGrid(this->grid);
    ui->openGLWidget->setCubeSize(ui->cubeSize->value());
}

void Widget::run()
{
    this->watcher.setFuture(
        QtConcurrent::run(
            [] (Widget *w)
            {
                cudaAlgorithmStep(&(w->grid), w->algorithmSteps);
            },
            this
        )
    );
}

void Widget::check()
{
    if(this->state == 1)
    {
        this->updateImage();
        this->updatePlots();
        this->run();
    }
}

void Widget::updateImage()
{
    // copy to temp
    cudaUpdateTempMatrix(&(this->grid));

    // Update OpenGL widget
    ui->openGLWidget->updateVBO();
    ui->openGLWidget->update();

//    // Update QImage widget
//    int imageWidth = this->grid.xSize;
//    int imageHeight = this->grid.ySize;

//    int widgetWidth = ui->graphicsView->width();
//    int widgetHeight = ui->graphicsView->height();

//    float imageRatio = imageWidth / (imageHeight + 0.0);
//    float widgetRatio = widgetWidth / (widgetHeight + 0.0);

//    int width;
//    int height;

//    if(imageRatio > widgetRatio)
//    {
//        width = widgetWidth;
//        height = width / imageRatio;
//    }
//    else
//    {
//        height = widgetHeight;
//        width = height * imageRatio;
//    }

//    QImage image = QImage(this->grid.xSize, this->grid.ySize, QImage::Format_RGB16);
//    for (int i = 0; i < this->grid.xSize; i++) {
//        for (int j = 0; j < this->grid.ySize; j++) {
////            int index = i + this->grid.ySize * j + this->grid.ySize * this->grid.zSize * this->coordZ;
//            int index = (i * this->grid.ySize + j) * this->grid.zSize + this->coordZ;
//            int value = this->grid.hostMatrix[index];
//            int r = this->colors[value][0];
//            int g = this->colors[value][1];
//            int b = this->colors[value][2];
//            image.setPixel(i, j, qRgb(r, g, b));
//        }
//    }

//    QGraphicsScene *oldScene = this->scene;
//    this->scene = new QGraphicsScene();
//    this->scene->addItem(
//        new QGraphicsPixmapItem(
//            QPixmap::fromImage(image).scaled(
//                width, height, Qt::KeepAspectRatio, Qt::FastTransformation
//            )
//        )
//    );
//    ui->graphicsView->setScene(this->scene);
//    delete oldScene;
}

void Widget::resizeEvent(QResizeEvent*)
{
    this->updateImage();
}

void Widget::updatePlots()
{
//    double e = this->grid->energy();
//    double m = qAbs(this->grid->magnetization());
//    double temperature = this->grid->temperature;

//    this->realStepList << ++this->counter;
//    this->realEnergyList << e;
//    this->realMagnetizationList << m;

//    this->energy += e;
//    this->energySquare += e * e;
//    this->magnetization += m;
//    this->magnetizationSquare += m * m;

//    int listSize = this->realStepList.size();
//    int calcSize = qMin(PLOT_SIZE, listSize);

//    if(listSize > calcSize)
//    {
//        this->realStepList.takeFirst();
//        this->chartEnergyList.takeFirst();
//        this->chartMagnetizationList.takeFirst();
//        this->chartHeatCapacityList.takeFirst();
//        this->chartMagneticSusceptibilityList.takeFirst();
//        e = this->realEnergyList.takeFirst();
//        m = this->realMagnetizationList.takeFirst();
//        this->energy -= e;
//        this->energySquare -= e * e;
//        this->magnetization -= m;
//        this->magnetizationSquare -= m * m;
//    }

//    double energy = this->energy / calcSize;
//    double energySquare = this->energySquare / calcSize;
//    double magnetization = this->magnetization / calcSize;
//    double magnetizationSquare = this->magnetizationSquare / calcSize;

//    // Расчёт теплоёмкости и магнитной восприимчивости
//    double heatCapacity = (energySquare - energy * energy) / (temperature * temperature);
//    double magneticSusceptibility = (magnetizationSquare - magnetization * magnetization) / temperature;

//    // Средние значения на один спин (удельные)
//    double gridTotal = this->grid->xSize * this->grid->ySize * this->grid->zSize;
//    this->chartEnergyList << energy / gridTotal;
//    this->chartMagnetizationList << magnetization / gridTotal;
//    this->chartHeatCapacityList << heatCapacity / gridTotal;
//    this->chartMagneticSusceptibilityList << magneticSusceptibility / gridTotal;

//    QVector<double> stepVector(this->realStepList.toVector());
//    QVector<double> energyVector(this->chartEnergyList.toVector());
//    QVector<double> magnetizationVector(this->chartMagnetizationList.toVector());
//    QVector<double> heatCapacityVector(this->chartHeatCapacityList.toVector());
//    QVector<double> magneticSusceptibilityVector(this->chartMagneticSusceptibilityList.toVector());

//    double stepMin, stepMax;
//    double energyMin, energyMax;
//    double magnetizationMin, magnetizationMax;
//    double heatCapacityMin, heatCapacityMax;
//    double magneticSusceptibilityMin, magneticSusceptibilityMax;
//    stepMin = stepMax = 0.0;
//    energyMin = energyMax = 0.0;
//    magnetizationMin = magnetizationMax = 0.0;
//    heatCapacityMin = heatCapacityMax = 0.0;
//    magneticSusceptibilityMin = magneticSusceptibilityMax = 0.0;

//    bool firstIteration = true;

//    for(int i = 0; i < stepVector.size(); i++)
//    {
//        double step = stepVector[i];
//        double energy = energyVector[i];
//        double magnetization = magnetizationVector[i];
//        double heatCapacity = heatCapacityVector[i];
//        double magneticSusceptibility = magneticSusceptibilityVector[i];

//        if(firstIteration)
//        {
//            stepMin = stepMax = step;
//            energyMin = energyMax = energy;
//            magnetizationMin = magnetizationMax = magnetization;
//            heatCapacityMin = heatCapacityMax = heatCapacity;
//            magneticSusceptibilityMin = magneticSusceptibilityMax = magneticSusceptibility;
//            firstIteration = false;
//        }
//        else
//        {
//            this->updateMinMax(step, stepMin, stepMax);
//            this->updateMinMax(energy, energyMin, energyMax);
//            this->updateMinMax(magnetization, magnetizationMin, magnetizationMax);
//            this->updateMinMax(heatCapacity, heatCapacityMin, heatCapacityMax);
//            this->updateMinMax(magneticSusceptibility, magneticSusceptibilityMin, magneticSusceptibilityMax);
//        }
//    }

//    this->updateMinMax(stepMin, stepMax);
//    this->updateMinMax(energyMin, energyMax);
//    this->updateMinMax(magnetizationMin, magnetizationMax);
//    this->updateMinMax(heatCapacityMin, heatCapacityMax);
//    this->updateMinMax(magneticSusceptibilityMin, magneticSusceptibilityMax);

//    // График удельной энергии
//    ui->customPlot1->clearGraphs();
//    ui->customPlot1->addGraph();
//    ui->customPlot1->graph(0)->setData(stepVector, energyVector);
//    ui->customPlot1->graph(0)->setScatterStyle(QCPScatterStyle(QCPScatterStyle::ssNone, Qt::darkBlue, 5.0));
//    ui->customPlot1->xAxis->setLabel("Шаг");
//    ui->customPlot1->yAxis->setLabel("Энергия");
//    ui->customPlot1->xAxis->setRange(stepMin, stepMax);
//    ui->customPlot1->yAxis->setRange(energyMin, energyMax);
//    ui->customPlot1->replot();

//    // График удельной намагниченности
//    ui->customPlot2->clearGraphs();
//    ui->customPlot2->addGraph();
//    ui->customPlot2->graph(0)->setData(stepVector, magnetizationVector);
//    ui->customPlot2->graph(0)->setScatterStyle(QCPScatterStyle(QCPScatterStyle::ssNone, Qt::darkBlue, 5.0));
//    ui->customPlot2->xAxis->setLabel("Шаг");
//    ui->customPlot2->yAxis->setLabel("Намагниченность");
//    ui->customPlot2->xAxis->setRange(stepMin, stepMax);
//    ui->customPlot2->yAxis->setRange(magnetizationMin, magnetizationMax);
//    ui->customPlot2->replot();

//    // График удельной теплоёмкости
//    ui->customPlot3->clearGraphs();
//    ui->customPlot3->addGraph();
//    ui->customPlot3->graph(0)->setData(stepVector, heatCapacityVector);
//    ui->customPlot3->graph(0)->setScatterStyle(QCPScatterStyle(QCPScatterStyle::ssNone, Qt::darkBlue, 5.0));
//    ui->customPlot3->xAxis->setLabel("Шаг");
//    ui->customPlot3->yAxis->setLabel("Теплоёмкость");
//    ui->customPlot3->xAxis->setRange(stepMin, stepMax);
//    ui->customPlot3->yAxis->setRange(heatCapacityMin, heatCapacityMax);
//    ui->customPlot3->replot();

//    // График магнитной восприимчивости
//    ui->customPlot4->clearGraphs();
//    ui->customPlot4->addGraph();
//    ui->customPlot4->graph(0)->setData(stepVector, magneticSusceptibilityVector);
//    ui->customPlot4->graph(0)->setScatterStyle(QCPScatterStyle(QCPScatterStyle::ssNone, Qt::darkBlue, 5.0));
//    ui->customPlot4->xAxis->setLabel("Шаг");
//    ui->customPlot4->yAxis->setLabel("Восприимчивость");
//    ui->customPlot4->xAxis->setRange(stepMin, stepMax);
//    ui->customPlot4->yAxis->setRange(magneticSusceptibilityMin, magneticSusceptibilityMax);
//    ui->customPlot4->replot();
}

void Widget::updateMinMax(double value, double& min, double& max)
{
    if(value < min) min = value;
    if(value > max) max = value;
}

void Widget::updateMinMax(double &min, double &max)
{
    if(qAbs(max - min) < std::numeric_limits<double>::epsilon())
    {
        min -= 0.001;
        max += 0.001;
    }
}
