Adopted
from [https://github.com/once-for-auto-driving/once_devkit/tree/master/once_eval](https://github.com/once-for-auto-driving/once_devkit/tree/master/once_eval)
on 08/19/2022.<br>
Note that the folder `submission_format` to test the original implementation is not contained here and you have to
obtain it from the repository above.

### 08/19/2022

* Replaced original IOU 3D calculation:

Original

```
iou_3d, _ = box3d_iou(corners_3d_ground, corners_3d_predict)
```

New

```
corners_3d_ground = torch.from_numpy(np.expand_dims(corners_3d_ground, axis=0).astype(np.float32))
corners_3d_predict = torch.from_numpy(np.expand_dims(corners_3d_predict, axis=0).astype(np.float32))
_, iou_3d = box3d_overlap(corners_3d_ground, corners_3d_predict)
iou_3d = iou_3d.numpy().item()
```

### 08/21/2022

* `prepare_a9_dataset_ground_truth()` and `prepare_predictions()` allow evaluation with A9-Dataset

Dataformat of predictions - one TXT file per frame with the content (one line per predicted object): class x y z l w h
rotation_z.<br>
Example

```
Car 16.0162 -28.9316 -6.45308 2.21032 3.74579 1.18687 2.75634
Car 17.926 -19.4624 -7.0266 1.03365 0.97037 0.435425 0.82854
```

### 08/29/2022

* Result shows how many objects per class there are in the predicitions as well as in the ground truth
* The precision formula is TP / (TP + FP), which leads to zero precision for TP = 0 and FP = 0. This case is corrected
  to precision = 100%
* The ground truth of the A9-Dataset is labeled with camera data, so there are objects contained in the ground truth
  without a single corresponding point in the LiDAR point cloud. To exclude objects with no or few points, you can
  specify how many minimum points there should be before a label is included.
* You can state that you want to include data from the Ouster LiDAR sensors only

Final result when evaluating the A9-Dataset R1 test set vs. itself:

```

|AP@50             |overall     |Occurrence (pred/gt)|
|Vehicle           |100.00      |2110/2110           |
|Pedestrian        |100.00      |32/32               |
|Bicycle           |100.00      |156/156             |
|mAP               |100.00      |2298/2298 (Total)   |
```