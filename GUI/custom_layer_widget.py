from PySide6.QtCore import Qt
from PySide6.QtGui import QAction
from PySide6.QtWidgets import QWidget, QHBoxLayout, QPushButton, QComboBox, QVBoxLayout, QLabel, QLineEdit, QMenu, \
    QMenuBar

from GUI.specific_parameters_widget import FloatingWidget
from GUI.utils import clearsubLayout






common_layers = [
    "None",
    "Dense",
    "Conv2D",
    "MaxPooling2D",
    "Flatten",
    "Dropout"
]

activation_functions = [
    "linear",
    "relu",
    "sigmoid",
    "tanh",
    "softmax",
    "softplus",
    "softsign",
    "selu",
    "elu",
    "exponential"
]
class LayerWidget(QWidget):
    def __init__(self):
        super(LayerWidget, self).__init__()
        self.mainLayout = QHBoxLayout()
        self.mainLayout.setAlignment(Qt.AlignTop)
        self.setLayout(self.mainLayout)
        self.setStyleSheet("background-color: #F67280;")
        self.setMinimumHeight(50)
        self.setMinimumWidth(100)
        self.firstl()

    def firstl(self):
        print("firstl")
        self.combo1 = QComboBox()
        self.combo1.addItems(common_layers)
        self.mainLayout.addWidget(self.combo1)
        self.combo1.currentTextChanged.connect(self.clear_layout)
        self.combo1.currentTextChanged.connect(self.choose_parameters)


    def choose_parameters2(self):
        print("chose parameters2")

        menubar = QMenuBar()
        # topMenu1 = menubar.addMenu('Convolutional')
        # topMenu2 = menubar.addMenu('Core')
        # topMenu3 = menubar.addMenu('Normalization')
        # topMenu4 = menubar.addMenu('Preprocessing')
        # topMenu5 = menubar.addMenu('Regularization')
        # topMenu6 = menubar.addMenu('Reshaping')
        # topMenu7 = menubar.addMenu('Pooling')
        # topMenu8 = menubar.addMenu('RNN')

        # 1
        ConvolutionalMenu = QMenu('Convolutional', self)
        menubar.addMenu(ConvolutionalMenu)
        actions1 = ["Conv1D","Convolution1D",    "Conv1DTranspose",    "Convolution1DTranspose",
    "Conv2D",    "Convolution2D",    "Conv2DTranspose",    "Convolution2DTranspose",    "Conv3D",    "Convolution3D",    "Conv3DTranspose",    "Convolution3DTranspose",    "DepthwiseConv1D",    "DepthwiseConv2D",    "SeparableConv1D",    "SeparableConvolution1D",    "SeparableConv2D",    "SeparableConvolution2D"
]
        for action in actions1:
            subAction = QAction(action, self)
            ConvolutionalMenu.addAction(subAction)

        # 2
        CoreMenu = QMenu('Core', self)
        menubar.addMenu(CoreMenu)
        core_layer_names = ["Activation", "Dense", "EinsumDense", "Embedding", "Lambda", "Masking", "ClassMethod", "InstanceMethod", "InstanceProperty", "SlicingOpLambda", "TFOpLambda"]

        for action in core_layer_names:
            subAction = QAction(action, self)
            CoreMenu.addAction(subAction)

        # 3
        NormalizationMenu = QMenu('Normalization', self)
        menubar.addMenu(NormalizationMenu)
        preprocessing_layer_names = [
            "CategoryEncoding", "Discretization", "HashedCrossing", "Hashing",
            "CenterCrop", "RandomBrightness", "RandomContrast", "RandomCrop", "RandomFlip",
            "RandomHeight", "RandomRotation", "RandomTranslation", "RandomWidth", "RandomZoom",
            "Rescaling", "Resizing", "IntegerLookup", "Normalization", "StringLookup",
            "TextVectorization"
        ]
        for action in preprocessing_layer_names:
            subAction = QAction(action, self)
            NormalizationMenu.addAction(subAction)

        self.mainLayout.addWidget(menubar)

    def choose_parameters(self):
        print("chose parameters")
        chosen_layer = self.combo1.currentText()
        print(chosen_layer)
        if chosen_layer == "Dense":
            self.DenseUI()
        elif chosen_layer == "Conv2D":
            self.Conv2DUi()
        elif chosen_layer == "MaxPooling2D":
            self.MaxPooling2DUI()
        elif chosen_layer == "Flatten":
            self.FlattenUI()
        elif chosen_layer == "Dropout":

            self.DropoutUI()
        else:
            pass
        if not chosen_layer=="None":
            self.add_end(self.mainLayout)


    def DenseUI(self):
        ll = ["units", "activ"]
        for l in ll:
            vertical = QVBoxLayout()
            vertical.setAlignment(Qt.AlignTop)
            label = QLabel(l)
            vertical.addWidget(label)
            if l =="units":
                self.firstwidget = QLineEdit("0")
                vertical.addWidget(self.firstwidget)
            elif l =="activ":
                self.secondwidget = QComboBox()
                self.secondwidget.addItems(activation_functions)
                vertical.addWidget(self.secondwidget)
            else:
                raise Exception("There should be case for that")


            self.mainLayout.addLayout(vertical)

        self.choose_parameters2()

    def Conv2DUi(self):
        ll = ["filters", "activ", "kernel_size"]
        for l in ll:
            vertical = QVBoxLayout()
            vertical.setAlignment(Qt.AlignTop)
            label = QLabel(l)
            if l == "filters":
                secondwidget = QLineEdit("0")
            elif l == "activ":
                secondwidget = QComboBox()
                secondwidget.addItems(activation_functions)
            elif l == "kernel_size":
                secondwidget = QLineEdit("0")
            else:
                raise Exception("There should be case for that")
            vertical.addWidget(label)
            vertical.addWidget(secondwidget)

            self.mainLayout.addLayout(vertical)


    def add_end(self, layout):
        button2 = QPushButton("specific")
        button2.setMaximumWidth(50)
        layout.addWidget(button2)
        button3 = QPushButton("delete")
        button3.setMaximumWidth(50)
        layout.addWidget(button3)
        button4 = QPushButton(u"insert \u2193")
        button4.setMaximumWidth(50)
        layout.addWidget(button4)

        button2.clicked.connect(self.handle_specific)
        button3.clicked.connect(self.handle_deletion)
        button4.clicked.connect(self.handle_insert)

    def handle_specific(self):
        print("handle specidifn")
        self.specific = FloatingWidget()
        self.specific.show()

    def handle_deletion(self):
        self.setParent(None)
        self.deleteLater()

    def handle_insert(self):
        parentLayout = self.parent().layout()  # Assuming the parent widget is set and has a layout
        index = parentLayout.indexOf(self)
        newWidget = LayerWidget()
        parentLayout.insertWidget(index + 1, newWidget)

    def FlattenUI(self):
        pass

    def DropoutUI(self):
        vertical = QVBoxLayout()
        vertical.setAlignment(Qt.AlignTop)
        label = QLabel("rate")
        secondwidget = QLineEdit("0")
        vertical.addWidget(label)
        vertical.addWidget(secondwidget)

        self.mainLayout.addLayout(vertical)


    def MaxPooling2DUI(self):
        ll = ["pool_size", "strides", "padding"]
        for l in ll:
            vertical = QVBoxLayout()
            vertical.setAlignment(Qt.AlignTop)
            label = QLabel(l)
            if l == "pool_size":
                secondwidget = QLineEdit("0")
            elif l == "strides":
                secondwidget = QLineEdit("0")
            elif l == "padding":
                secondwidget = QComboBox()
                secondwidget.addItems(["valid", "same"])
            else:
                raise Exception("There should be case for that")
            vertical.addWidget(label)
            vertical.addWidget(secondwidget)

            self.mainLayout.addLayout(vertical)

    def clear_layout(self):
        if self.mainLayout is not None:
            for i in range(self.mainLayout.count() - 1, -1, -1):
                if not i == 0:
                    item = self.mainLayout.itemAt(i)
                    if item.widget():
                        item.widget().deleteLater()
                    elif item.layout():
                        clearsubLayout(item)
                        self.mainLayout.removeItem(item)
                    else:
                        raise Exception("something wrong in deleting layouts")

    def return_values(self):
        return {"layer_type": self.combo1.currentText(), "units":self.firstwidget.text(), "activation":self.secondwidget.currentText()}
