# CTracker (ECCV2020 Spotlight)

my implementation  of **Chained-Tracker** as described in [Chained-Tracker: Chaining Paired Attentive Regression Results for End-to-End Joint Multiple-Object Detection and Tracking](https://arxiv.org/abs/2007.14557).
 
The introduction video of CTracker is uploaded to [Youtube](https://www.youtube.com/watch?v=UovwAgKys88).

## Video demos on MOT challenge test set
<img src="demos/MOT17-03.gif" width="400"/>   <img src="demos/MOT17-07.gif" width="400"/>
<img src="demos/MOT17-08.gif" width="400"/>   <img src="demos/MOT17-12.gif" width="400"/>

## Organize MOT17 dataset
MOT17 dataset can be downloaded at [MOTChallenge](https://motchallenge.net/data/MOT17Det/).

We uses two CSV files to organize the MOT17 dataset: one file containing annotations and one file containing a class name to ID mapping. 

We provide the two CSV files for MOT17 with codes in the CTRACKER_ROOT/data, you should copy them to MOT17_ROOT before starting training. 

### Dataset structures:
```
MOT17_ROOT/
        |->train/
        |    |->MOT17-02/
        |    |->MOT17-04/
        |    |->...
        |->test/
        |    |->MOT17-01/
        |    |->MOT17-03/
        |    |->...
        |->train_annots.csv
        |->train_labels.csv
```
MOT17_ROOT is your path of the MOT17 Dataset.


### Annotations format
The CSV file with annotations should contain one annotation per line.
Images with multiple bounding boxes should use one row per bounding box.
Note that indexing for pixel values starts at 0.
The expected format of each line is:
```
path/to/image.jpg,id,x1,y1,x2,y2,class_name
```

The MOT17 CSV file can be generated by generate_csv.py:
```
python generate_csv.py
```
You can modify this script to handle other datasets.

### Class mapping format
The class name to ID mapping file should contain one mapping per line.
Each line should use the following format:
```
class_name,id
```

Indexing for classes starts at 0.
Do not include a background class as it is implicit.

For example:
```
person,0
```

## Training

The network can be trained using the `train.py` script. For training on MOT17, use

```
CUDA_VISIBLE_DEVICES=0 python train.py --root_path MOT17_ROOT --model_dir ./ctracker/ --depth 50
```
By default, testing will start immediately after training finished.

## Testing

A trained model is available at [Google Drive](https://drive.google.com/file/d/1-5f-3QwcDoFL6b3_81tcsYTWsU43aBaz/view?usp=sharing)/[Tencent Weiyun](https://share.weiyun.com/KgWrWCv3), run the following commands to start testing:

```
CUDA_VISIBLE_DEVICES=0 python test.py --dataset_path MOT17_ROOT --model_dir ./trained_model/
```


## Acknowledgements

- Part of codes are borrowed from the [pytorch retinanet implementation](htt
ps://github.com/yhenon/pytorch-retinanet)
- The NMS module used is from the [simpledet](https://github.com/TuSimple/simpledet)


## Citing CTracker

If you find CTracker is useful in your project, please consider citing us:

```BibTeX
@inproceedings{peng2020ctracker,
  title={Chained-Tracker: Chaining Paired Attentive Regression Results for End-to-End Joint Multiple-Object Detection and Tracking},
  author={Peng, Jinlong and Wang, Changan and Wan, Fangbin and Wu, Yang and Wang, Yabiao and Tai, Ying and Wang, Chengjie and Li, Jilin and Huang, Feiyue and Fu, Yanwei},
  booktitle={Proceedings of the European Conference on Computer Vision},
  year={2020},
}
```


