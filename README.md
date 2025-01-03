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

![diagram](./images/treatment_plant_diagram.jpg)

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

## Input Features

| Output values | Input features 1 | Input features 2 | Input features 3 | Input features 4 | Input features 5 | Input features 6 | Input features 7 | Input features 8 | Input features 9 |
|---------------|------------------|------------------|------------------|------------------|------------------|------------------|------------------|------------------|------------------|
|               | Feature 1        | Feature 2        | Feature 3        | Feature 4        | Feature 5        | Feature 6        | Feature 7        | Feature 8        | Feature 9        |




## Neural Network Structure

The chosen neural network architecture will be simple to facilitate easy implementation on a Siemens S1500 PLC. I will experiment with 9 input features, a hidden layer with 6 neurons using a ReLU (Rectified Linear Unit) activation function, which is straightforward to implement. The output layer will consist of a single neuron without an activation function, allowing for continuous output across the full range.

This design aims to be both efficient and compatible with the constraints of the PLC system, ensuring the neural network can be deployed and run in a real-time environment.

## What next?

How could your project grow and become something even more? What kind of skills, what kind of assistance would you  need to move on? 


## Acknowledgments

* list here the sources of inspiration 
* do not use code, images, data etc. from others without permission
* when you have permission to use other people's materials, always mention the original creator and the open source / Creative Commons licence they've used
  <br>For example: [Sleeping Cat on Her Back by Umberto Salvagnin](https://commons.wikimedia.org/wiki/File:Sleeping_cat_on_her_back.jpg#filelinks) / [CC BY 2.0](https://creativecommons.org/licenses/by/2.0)
* etc
