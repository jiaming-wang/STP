#  STP (Updating)
Spatial-temporal Pooling for Action Recognition in Videos

### *Ferryboat-4*

The *Ferryboat-4* includes 4 action classes: *Inshore*, *Offshore*, *Traffic*, and *Negative*. 

----------
![image](/img/ferryboat.jpg)
----------
Image 1. Sample RGB and optical flow frames of the *Ferryboat-4* dataset. We definde *Inshore* of a ferry as the process from appearing in carmera to berthing, while *Offshore* is contrary. *Traffic* mainly includes the movement of pedestrians and vehicles, such as boarding, disembarking and others. Therefore, other scenes are divided into negative samples.

We defined *Inshore* of a ferry as the process from appearing in camera to berthing (as shown in(A)), while *Offshore* is contrary (as shown in (B)). *Traffic* mainly includes the movement of pedestrians and vehicles, such as boarding, disembarking and others. To ensure the diversity of scenes, it includes different vehicles (motorcycle, tricycle, minibus, bicycle, and electric motorcar). Therefore, other scenes are divided into negative samples, for example standing water, stationary ferryboat and others. 

Table 2. Comparison with existing methods on *Ferryboat-4*. 
|Categories|Data|
|:---:|:---:|
|Actions | 4 |
|Clips | 431 |
|Total Duration | 147.7 minutes |
|Frame Rate | 10 FPS/s |
|Reolution | 1920 × 1080 |
|Audio | No |

![image](/img/acc_epoch.jpg)

Image 2. Training accuracy for different methods. (A) The results on RGB-*Ferryboat-4*. (B) The results on flow-*Ferryboat-4*.

Table 1. Comparison with existing methods on *Ferryboat-4*. 
||C3D|R2+1D|R3D|I3D|P3D|STP|
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
|Flow | 49.49 |49.48| 56.32| 58.71| 50.64| 65.44|
|RGB | 61.64 |63.62| 62.04| 63.81| 62.68| 66.91|
|Two-stream | 65.05 |63.87| 62.19| 66.05| 64.23| 67.75|
