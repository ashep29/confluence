# Confluence
Official Python implementation of the Confluence and Confluence NMS algorithms as described in [Confluence: A Robust Non-IoU Alternative to Non-Maxima Suppression in Object Detection](https://arxiv.org/abs/2012.00257)

![confluence](https://user-images.githubusercontent.com/39542635/135226907-e74ebb57-284e-4090-9923-53f49468c987.png)

Confluence and Confluence-NMS are non-IoU alternatives to Greedy and Soft-NMS and their variants. They rely on a Manhattan Distance inspired proximity measure to retain true positives and suppress false positives. Proximity represents the sum of the vertical and horizonal distances between the upper left and lower right coordinates to two bounding boxes, as shown below:

![Proximity_description_resized](https://user-images.githubusercontent.com/39542635/120128561-a4c60d00-c205-11eb-83a2-630c012a672c.PNG)


Proximity is a much more stable, consistent predictor of localization than IoU due to its purely linear relationship with bounding box overlap, as illustrated by the graph below. In contrast, IoU is not linear, increasing the sensitivity of the NMS IoU threshold to slight variations in overlap. This makes IoU more susceptible to inconsistent, suboptimal performance, often resulting in the retention of false positives and suppression of true positives.

![IoU v Proximity_resized](https://user-images.githubusercontent.com/39542635/120128567-a8f22a80-c205-11eb-8061-397f6b7ffbdd.png)

