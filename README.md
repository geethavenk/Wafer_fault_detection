# Wafer Fault Prediction

### Brief

In electronics, a wafer (also known as a slice or substrate) is a thin slice of semiconductor material, such as crystalline silicon (c-Si), used for the fabrication of integrated circuits and in photovoltaics for the manufacture of solar cells. The wafer acts as the foundation for microelectronic devices built in and upon it.

Wafers undergo numerous microfabrication processes, including doping, ion implantation, etching, thin-film deposition of various materials, and photolithographic patterning. Finally, the individual microcircuits are separated by wafer dicing and packaged as integrated circuits.

## Problem Statement

### Data
The dataset consists of wafers' data, which includes readings from hundreds of sensors embedded in the wafers.

### Problem
Wafers are primarily used to manufacture solar cells and are often located in remote areas in bulk. They play a fundamental role in photovoltaic power generation, which converts sunlight directly into electrical energy. The production of these wafers requires high technology and precision.

Identifying faulty wafers is crucial to maintain efficiency and reduce costs. Traditionally, this task requires manual inspection. When a fault is suspected, the wafer must be manually examined from scratch. This process disrupts production as all nearby wafers must be stopped, resulting in significant time, manpower, and financial losses, especially if the suspicion of a fault is incorrect.

### Solution
By passing the data collected from wafers through a machine learning pipeline, we can accurately determine whether a wafer is faulty. This automated approach eliminates the need for manual inspection, thereby reducing costs and improving efficiency.

### Project Structure
```.
├── src
│   ├── components
│   ├── pipelines
│   ├── utils.py
│   └── logger.py
├── data
│   ├── raw
│   └── processed
├── artifacts
├── notebooks
├── app.py
├── requirements.txt
└── README.md
```

### Usage

#### 1. Clone the repository:
```.
git clone https://github.com/yourusername/wafer-fault-prediction.git
```
#### 2. Navigate to the project directory:
```.
cd wafer-fault-prediction
```
#### 3. Install the required dependencies:
```.
pip install -r requirements.txt
```
#### 4. Run the application:
```.
streamlit run main.py
```

### Contributing
Contributions are welcome! Please feel free to submit a Pull Request.






