<!-- This is the markdown template for the final project of the Building AI course, 
created by Reaktor Innovations and University of Helsinki. 
Copy the template, paste it to your GitHub README and edit! -->

# Smart-Chlorination
Final project for the Building AI course


## Summary

This project aims to predict the pulse rate of a chlorine dosing pump in a drinking water treatment plant based on water characteristics. An AI-based model will be developed to calculate the required pulse frequency with precision.

The model will be integrated into a Siemens S1500 PLC to enable real-time data processing and automated pump control, ensuring efficient and accurate chlorine dosing.


## Background

In water treatment plants, maintaining the free chlorine level is often controlled using a classic PID (Proportional-Integral-Derivative) controller. However, the process requires a contact time between the chlorine dosing and the measurement of free chlorine in the water to ensure proper disinfection. This mandatory contact time introduces a delay in the control loop, leading to potential challenges such as instability, oscillations, or overshooting in the chlorine dosing process. These issues can compromise water quality and dosing efficiency.


## How is it used?

The aim of this project is to implement an artificial intelligence (AI)-based solution to precisely predict the pulse frequency (rate pulse) of sodium hypochlorite dosing pumps in the water treatment plant.
Although the PID controller will remain in a closed loop, its main function will be compensatory, making secondary adjustments to the free chlorine level. The accuracy of control will primarily depend on the predictive AI, which will reduce fluctuations and oscillations in the system and improve the overall stability of the dosing process.

It is possible to use linear regression for this task, but a small neural network can account for the breakpoint in chlorination and other non-linearities.



## Data sources and AI methods
Where does your data come from? Do you collect it yourself or do you use data collected by someone else?
If you need to use links, here's an example:
[Twitter API](https://developer.twitter.com/en/docs)

| Syntax      | Description |
| ----------- | ----------- |
| Header      | Title       |
| Paragraph   | Text        |

## Challenges

What does your project _not_ solve? Which limitations and ethical considerations should be taken into account when deploying a solution like this?

## What next?

How could your project grow and become something even more? What kind of skills, what kind of assistance would you  need to move on? 


## Acknowledgments

* list here the sources of inspiration 
* do not use code, images, data etc. from others without permission
* when you have permission to use other people's materials, always mention the original creator and the open source / Creative Commons licence they've used
  <br>For example: [Sleeping Cat on Her Back by Umberto Salvagnin](https://commons.wikimedia.org/wiki/File:Sleeping_cat_on_her_back.jpg#filelinks) / [CC BY 2.0](https://creativecommons.org/licenses/by/2.0)
* etc
