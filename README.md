<!-- This is the markdown template for the final project of the Building AI course, 
created by Reaktor Innovations and University of Helsinki. 
Copy the template, paste it to your GitHub README and edit! -->

# Smart-Chlorination
Final project for the Building AI course


## Summary

Predict the pulse frequency required for sodium hypochlorite dosing pumps in a drinking water treatment plant to achieve the desired free chlorine level after 10 minutes of contact, considering the characteristics of the incoming water.


## Background

In water treatment plants, using a classic PID (Proportional-Integral-Derivative) controller to regulate the free chlorine level is common. However, a delay between dosing and measurement can cause several problems:

Response Delay: The PID controller may react incorrectly due to the delay. If the system takes 10 minutes to measure the effect of dosing, the controller might overcompensate or undercompensate, causing fluctuations in the chlorine level.

Oscillations: The delay can induce oscillations in the system. The PID controller might adjust dosing based on outdated data, leading to a cycle of over-adjustments and under-adjustments.

Inaccurate Control: The precision of controlling the chlorine level is compromised. The goal is to maintain a constant free chlorine level, but with a significant delay, achieving precise and stable control is difficult.

Risk of Under-dosing or Over-dosing: A prolonged delay can result in unsafe chlorine levels. Under-dosing may not completely eliminate pathogens, while over-dosing can lead to chlorine levels that are harmful to health and the environment.

Frequent Manual Adjustments: Operators may need to intervene manually more often to correct chlorine levels, increasing workload and the risk of human error.

To mitigate these issues, strategies such as using predictive models to anticipate the effect of dosing before it is measured, or adjusting the PID parameters to account for the delay, can be implemented.


## How is it used?

The aim of this project is to implement an artificial intelligence (AI)-based solution to precisely predict the pulse frequency (rate pulse) of sodium hypochlorite dosing pumps in the water treatment plant. This solution uses predictive models to anticipate the impact of dosing before it is measured, considering the characteristics of the incoming water and the inherent feedback delay.
Although the PID controller will remain in a closed loop, its main function will be compensatory, making secondary adjustments to the free chlorine level. The accuracy of control will primarily depend on the predictive AI, which will reduce fluctuations and oscillations in the system and improve the overall stability of the dosing process.
The intention of this project is to improve the efficiency, precision, and stability of chlorine level control in treated water, ensuring a faster and more adaptive response to changing water conditions, optimizing the use of chemicals, and minimizing manual intervention and the risk of errors.

Describe the process of using the solution. In what kind situations is the solution needed (environment, time, etc.)? Who are the users, what kinds of needs should be taken into account?

Images will make your README look nice!
Once you upload an image to your repository, you can link link to it like this (replace the URL with file path, if you've uploaded an image to Github.)
![Cat](https://upload.wikimedia.org/wikipedia/commons/5/5e/Sleeping_cat_on_her_back.jpg)

If you need to resize images, you have to use an HTML tag, like this:
<img src="https://upload.wikimedia.org/wikipedia/commons/5/5e/Sleeping_cat_on_her_back.jpg" width="300">

This is how you create code examples:
```
def main():
   countries = ['Denmark', 'Finland', 'Iceland', 'Norway', 'Sweden']
   pop = [5615000, 5439000, 324000, 5080000, 9609000]   # not actually needed in this exercise...
   fishers = [1891, 2652, 3800, 11611, 1757]

   totPop = sum(pop)
   totFish = sum(fishers)

   # write your solution here

   for i in range(len(countries)):
      print("%s %.2f%%" % (countries[i], 100.0))    # current just prints 100%

main()
```


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
