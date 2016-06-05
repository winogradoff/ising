
#include "widget.h"
#include "ui_widget.h"
#include <QMessageBox>
#include <QGraphicsPixmapItem>
#include <QGraphicsItem>
#include <QTime>
#include <QDebug>
#include <QImage>

Widget::Widget(QWidget* parent)
    : QWidget(parent)
    , ui(new Ui::Widget)
{
    VisualWGT = NULL;
    PlotWGT = NULL;
    ui->setupUi(this);
    connect(&(this->watcher), &QFutureWatcher<void>::finished, this, &Widget::check);

    this->grid.randomStates = NULL;
    this->grid.deviceMatrix = NULL;

    this->state = 0;

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

void Widget::on_cubeSize_valueChanged(int value)
{
    if (this->VisualWGT != NULL)
    {
        this->VisualWGT->OpenGLWGT->setCubeSize(value);
        this->VisualWGT->OpenGLWGT->update();
    }
}

void Widget::on_newButton_clicked()
{
    this->newGrid();
    this->updateVisual();
    this->updatePlots();
}

void Widget::on_startStopButton_clicked()
{
    if (this->state == 0)
    {
        //        if(this->VisualWGT==NULL)
        //        {
        //            on_visualButton_clicked();
        //        }
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
            break;

        case DIM_2:
            ui->gridXLabel->setVisible(true);
            ui->gridXSize->setVisible(true);
            ui->gridYLabel->setVisible(true);
            ui->gridYSize->setVisible(true);
            ui->gridZLabel->setVisible(false);
            ui->gridZSize->setVisible(false);
            break;

        case DIM_3:
            ui->gridXLabel->setVisible(true);
            ui->gridXSize->setVisible(true);
            ui->gridYLabel->setVisible(true);
            ui->gridYSize->setVisible(true);
            ui->gridZLabel->setVisible(true);
            ui->gridZSize->setVisible(true);
            break;
    }
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

    switch (this->grid.dimension)
    {
        case DIM_1:
            this->grid.ySize = this->grid.zSize = 1;
            break;

        case DIM_2:
            this->grid.zSize = 1;
            break;

        case DIM_3:
            break;
    }

    cudaFreeGrid(&(this->grid));
    cudaInitGrid(&(this->grid));

    if (this->VisualWGT != NULL)
    {
        this->VisualWGT->OpenGLWGT->setGrid(this->grid);
        this->VisualWGT->OpenGLWGT->setCubeSize(ui->cubeSize->value());
    }

    if (this->PlotWGT != NULL)
    {
        this->PlotWGT->setGrid(this->grid);
    }
}

void Widget::run()
{
    this->watcher.setFuture(QtConcurrent::run(
        [](Widget* w)
        {
            cudaAlgorithmStep(&(w->grid), w->algorithmSteps);
        },
        this));
}

void Widget::check()
{
    this->updateVisual();
    this->updatePlots();
    if (this->state == 1)
    {
        this->run();
    }
}

void Widget::updateVisual()
{
    if (this->VisualWGT != NULL)
    {
        this->VisualWGT->OpenGLWGT->updateVBO();
        this->VisualWGT->OpenGLWGT->update();
    }
}

void Widget::updatePlots()
{
    if (this->PlotWGT != NULL)
    {
        this->PlotWGT->updatePlots(cudaEnergy(&(this->grid)), cudaMagnetization(&(this->grid)));
    }
}

void Widget::closeEventVisualWGT(QCloseEvent*)
{
    if (VisualWGT != NULL)
    {
        VisualWGT->close();
        delete VisualWGT;
        VisualWGT = NULL;
    }
}

void Widget::on_visualButton_clicked()
{
    if (VisualWGT == NULL)
    {
        ui->visualButton->setEnabled(false);
        ui->plotButton->setEnabled(false);
        VisualWGT = new VisualWidget();
        VisualWGT->show();
        connect(VisualWGT, SIGNAL(closedSignal()), this, SLOT(on_visualButton_clicked()));
        this->VisualWGT->OpenGLWGT->setGrid(this->grid);
        this->VisualWGT->OpenGLWGT->setCubeSize((ui->cubeSize->value()));
        this->VisualWGT->OpenGLWGT->update();
    }
    else if (VisualWGT != NULL)
    {
        VisualWGT->close();
        VisualWGT = NULL;
        ui->visualButton->setEnabled(true);
        ui->plotButton->setEnabled(true);
    }
}

void Widget::closeEventPlotWGT(QCloseEvent*)
{
    if (PlotWGT != NULL)
    {
        PlotWGT->close();
        delete PlotWGT;
        PlotWGT = NULL;
    }
}

void Widget::on_plotButton_clicked()
{
    if (PlotWGT == NULL)
    {
        ui->visualButton->setEnabled(false);
        ui->plotButton->setEnabled(false);
        PlotWGT = new PlotWidget();
        PlotWGT->setGrid(this->grid);
        PlotWGT->show();
        connect(PlotWGT, SIGNAL(closedSignal()), this, SLOT(on_plotButton_clicked()));
    }
    else if (PlotWGT != NULL)
    {
        PlotWGT->close();
        delete PlotWGT;
        PlotWGT = NULL;
        ui->visualButton->setEnabled(true);
        ui->plotButton->setEnabled(true);
    }
}
