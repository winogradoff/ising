#include "visualwidget.h"
#include "ui_visualwidget.h"

VisualWidget::VisualWidget(QWidget* parent)
    : QWidget(parent)
    , ui(new Ui::VisualWidget)
{
    ui->setupUi(this);
    OpenGLWGT = ui->openGLWidget;
}

VisualWidget::~VisualWidget()
{
    delete ui;
}

void VisualWidget::closeEvent(QCloseEvent* e)
{
    QWidget::closeEvent(e);
    emit closedSignal();
}
