# Confluence
Python implementation of the Confluence and Confluence NMS algorithms.

Confluence and Confluence-NMS are non-IoU alternatives to Greedy and Soft-NMS and their variants. They rely on a Manhattan Distance inspired proximity measure to retain true positives and suppress false positives. Proximity represents the sum of the vertical and horizonal distances between the upper left and lower right coordinates to two bounding boxes, as shown below:
![Proximity_description](https://user-images.githubusercontent.com/39542635/120127863-e6ee4f00-c203-11eb-83b5-4dcfcfbc0b23.PNG)

Proximity is a much more stable, consistent predictor of localization than IoU due to its purely linear relationship with bounding box overlap, as illustrated by the graph below. In contrast, IoU is not linear, increasing the sensitivity of the NMS IoU threshold to slight variations in overlap. This makes IoU more susceptible to inconsistent, suboptimal performance, often resulting in the retention of false positives and suppression of true positives.
![IoU v Proximity](https://user-images.githubusercontent.com/39542635/120128025-69770e80-c204-11eb-8695-6d43e9a3d109.png)
![IoU v Proximity_resized](https://user-images.githubusercontent.com/39542635/120128423-4436d000-c205-11eb-8233-6e8b4a6deb9d.png)

