<!-- This is the markdown template for the final project of the Building AI course, 
created by Reaktor Innovations and University of Helsinki. 
Copy the template, paste it to your GitHub README and edit! -->

# Smart-Chlorination
Final project for the Building AI course


## Summary

This project focuses on predicting the pulse rate of a chlorine dosing pump in a drinking water treatment plant based on water characteristics. An AI model will be developed to calculate the required pulse frequency.

The model will be deployed on a Siemens S1500 PLC to facilitate real-time data processing and automated pump control.

## Background

In water treatment plants, maintaining free chlorine levels is commonly managed using a classic PID (Proportional-Integral-Derivative) controller. However, the process requires a specific contact time between chlorine dosing and the measurement of free chlorine to ensure proper disinfection. This introduces a delay in the control loop, which can result in challenges such as instability, oscillations, or overshooting, potentially compromising water quality and dosing efficiency.

This project aims to implement an artificial intelligence based solution to predict the pulse frequency of sodium hypochlorite dosing pumps. While the PID controller will remain part of the closed loop, its role will shift to compensatory adjustments, focusing on fine-tuning the free chlorine level. By relying on predictive AI, the system will reduce fluctuations and enhance the stability of the dosing process.

Although linear regression could be used for this task, a small neural network is better suited to capture the breakpoint in chlorination and address non-linear behaviors effectively.


## The Treatment Plant

![treatment_plant_diagram](./images/treatment_plant_diagram.png)

The treatment process begins with the dosing of coagulant, followed by filtration of the water. After that, the water undergoes ultraviolet (UV) treatment, and finally, it is chlorinated.

We monitor several parameters throughout the process, including:

- Turbidity of raw water
- PPM of coagulant
- UV percentage
- Which dosing pump is operating
- Hypochlorite tank level
- pH levels
- Free chlorine
- Chlorine setpoint
- Turbidity of treated water
- Flow rate
- Free chlorine treated water

## Data sources
We have an extensive dataset, with measurements recorded every 2 minutes over several years, including the parameters mentioned earlier. However, careful filtering is required to select only the data that can effectively train the neural network.

### Data Filtering
1. Initial Data Filtering:
At first, I filtered the data by discarding entries where the difference between the setpoint (desired free chlorine) and the measurement (measured free chlorine) was greater than 1% of the setpoint value. I then trained the neural network with the remaining data (as I will explain later). However, the data wasn't good enough, which led to convergence problems during training.

2. Alternative Data Filtering Approach:
To address these problems, I tried a different filtering method. Instead of discarding only the data where the difference between the setpoint and the measured value was greater than 1%, I also removed all entries where such differences occurred in the 5 readings before and after the current entry. This approach aimed to retain only the data that more accurately captured the relationship between water parameters and the pulse rate of the pumps.

## Data Format Description
The dataset is stored in a .csv file with a total of 8,463 rows, including the header row. The structure of the file is as follows:

| Pulses | TankLvl | ClSet | Pump | UV%   | Flow  | InfTurb | EffTurb | pH   | CoagPPM |
|--------|---------|-------|------|-------|-------|---------|---------|------|---------|
| 53.55  | 64.87   | 0.7   | 1    | 31.37 | 88.28 | 1.81    | 0.88    | 9.21 | 0       |
| 58.16  | 78.75   | 0.7   | 1    | 25.93 | 96.72 | 0.61    | 0.32    | 8.8  | 0       |
| 44.44  | 64.19   | 0.7   | 2    | 35.08 | 97.81 | 1.06    | 0.57    | 9.06 | 0       |
| 56.36  | 69.62   | 0.7   | 1    | 32.24 | 90.47 | 0.69    | 0.39    | 8.79 | 0       |
| ...    | ...     | ...   | ...  | ...   | ...   | ...     | ...     | ...  | ...     |

- **The first column** (Output) represents the pump pulses This is the target variable the network is trained to predict.
- **The next 9 columns** (Feature1 to Feature9) correspond to the input features:
  - **Feature1:** Hypo. tank level
  - **Feature2:** Chlorine setpoint
  - **Feature3:** Dosing pump
  - **Feature4:** UV percentage
  - **Feature5:** Flow
  - **Feature6:** Influent turbidity
  - **Feature7:** Effluent turbidity
  - **Feature8:** pH
  - **Feature9:** Coagulant PPM
 
The dataset file is located in the **data** folder under the name **water_treatment_data.csv**.

## Neural Network Structure
The chosen neural network architecture will be simple to facilitate easy implementation on a Siemens S1500 PLC. I will experiment with 9 input features, a hidden layer with 6 neurons using a ReLU (Rectified Linear Unit) activation function, which is straightforward to implement. The output layer will consist of a single neuron without an activation function, allowing for continuous output across the full range.

This design aims to be both efficient and compatible with the constraints of the PLC system, ensuring the neural network can be deployed and run in a real-time environment.

![NeuralNetworkDiagram](./images/NeuralNetworkDiagram.png)

## Supervised Training of the Model
The model training process will be carried out using Python, leveraging its powerful libraries such as TensorFlow or Keras for building and training the neural network. During this process, I have used the assistance of an AI tool (ChatGPT) to guide me through the coding.

The program must perform the following tasks:
1. Data Preparation
   - Reads data from a CSV file, separates input features (9 columns) and the target variable (1 column).
   - Converts data to numeric and standardizes features using StandardScaler to improve model performance.
2. Model Creation
   - Defines a simple neural network with:
     - 1 hidden layer (6 neurons, ReLU activation).
     - 1 output neuron (linear activation for regression).
   - Uses adam optimizer, mean squared error (MSE) as the loss function, and mean absolute error (MAE) as a metric.
3. Training
   - Splits data into training (80%) and validation (20%) sets using scikit-learn's train_test_split.
   - Trains the model with TensorFlow/Keras, applying backpropagation and Early Stopping to avoid overfitting.
   - Tested multiple configurations and techniques to optimize performance, including:
     - Dropout layers.
     - K-fold Cross-Validation
     - Different dataset splits.
     - Tuning the number of epochs and batch size.
    - After evaluating the results, the final configuration was selected based on the best balance between validation loss and R<sup>2</sup> score.   
5. Evaluation
    - Assesses the model on the validation set using MSE, MAE, and R<sup>2</sup>.
6. Visualization
   - Creates graphs to evaluate performance, including loss curves, scatter plots of predictions vs. actual values, residual distributions, and line plots.
7. Exporting Results
   - Extracts weights, biases, and scaling parameters (mean and standard deviation) from the model and displays them in a format that can be directly copied and pasted into a TIA Portal Data Block.

The program is available in the **src** folder under the name **train_neural_network.py**.

## Implementing the Neural Network in a Siemens S7-1500 PLC


- [Hidden Neuron](https://github.com/lmpipaon/Smart-Chlorination/blob/main/PLC/HIDDEN_NEURON.pdf)
- [Neural Network](https://github.com/lmpipaon/Smart-Chlorination/blob/main/PLC/NEURAL_NETWORK.pdf)
- [Neural Network Initialization](https://github.com/lmpipaon/Smart-Chlorination/blob/main/PLC/NEURAL_NETWORK_INITIALIZATION.pdf)
- [Neural Network Weights Biases Normalization](https://github.com/lmpipaon/Smart-Chlorination/blob/main/PLC/NEURAL_NETWORK_WEIGHTS_BIASES_NORMALIZATION.pdf)
- [Normalization](https://github.com/lmpipaon/Smart-Chlorination/blob/main/PLC/NORMALIZATION.pdf)
- [Output Neuron](https://github.com/lmpipaon/Smart-Chlorination/blob/main/PLC/OUTPUT_NEURON.pdf)

