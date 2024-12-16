# Prepare Datasets for SCSD
**:star2: The dataset preparation refers to [HGFormer](https://github.com/dingjiansw101/HGFormer/blob/main/datasets/README.md).**

A dataset can be used by accessing [DatasetCatalog](https://detectron2.readthedocs.io/modules/data.html#detectron2.data.DatasetCatalog) for its data, or [MetadataCatalog](https://detectron2.readthedocs.io/modules/data.html#detectron2.data.MetadataCatalog) for its metadata (class names, etc). This document explains how to setup the builtin datasets so they can be used by the above APIs. [Use Custom Datasets](https://detectron2.readthedocs.io/tutorials/datasets.html) gives a deeper dive on how to use `DatasetCatalog` and `MetadataCatalog`, and how to add new datasets to them.

SCSD has builtin support for a few datasets. The datasets are assumed to exist in a directory specified by the environment variable DETECTRON2_DATASETS. Under this directory, detectron2 will look for datasets in the structure described below, if needed.
```
$DETECTRON2_DATASETS/
  cityscapes/
  acdc/
  mapillary/
  bdd/
  gta/
  synthia/
```

You can set the location for builtin datasets by `export DETECTRON2_DATASETS=/path/to/datasets`.
If left unset, the default is `./datasets` relative to your current working directory.


## Expected dataset structure for [Cityscapes](https://www.cityscapes-dataset.com/downloads/):
```
cityscapes/
  gtFine/
    train/
      aachen/
        color.png, instanceIds.png, labelIds.png, polygons.json,
        labelTrainIds.png
      ...
    val/
    test/
  leftImg8bit/
    train/
    val/
    test/
```

Install cityscapes scripts by:
```
pip install git+https://github.com/mcordts/cityscapesScripts.git
```

Note: to create labelTrainIds.png, first prepare the above structure, then run cityscapesescript with:
```
CITYSCAPES_DATASET=/path/to/abovementioned/cityscapes python cityscapesscripts/preparation/createTrainIdLabelImgs.py
```

## Expected dataset structure for [ACDC](https://acdc.vision.ee.ethz.ch/download):
```
acdc/
    rgb_anon/
        fog/
            test/
        night/
            test/
        rain/
            test/
        snow/
            test/
        all/
            test/
    
```
You should create the folder of ```all``` and copy test images of all types to ```all/test```

## Expected dataset structure for [Mapillary](https://www.mapillary.com/dataset/vistas):
```
mapillary/
    training/
        images/
        labels
    validation/
        images/
        labels/
    testing/
        images/
        labels/
    labels_detectron2/
        training/
        validation/
```
Run `python datasets/prepare_mapillary_sem_seg.py`, to map the mapillary labels to the Cityscapes labels


## Expected dataset structure for [BDD](https://doc.bdd100k.com/download.html):
```
bdd/
    images/
        10k/
          train/
          val/
    labels/
        sem_seg/
          masks/
            train/
            val/
```

## Expected dataset structure for [GTAV](https://download.visinf.tu-darmstadt.de/data/from_games/):
```
gta/
    images/
        train/
        valid/
        test/
    labels/
        train/
        valid/
        test/
    labels_detectron2/
        train/
        valid/
        test/
```
Downlaod the GTA from https://download.visinf.tu-darmstadt.de/data/from_games/

Then unzip the images and labels.

We split the dataset following [RobustNet](https://github.com/shachoi/RobustNet)
``` 
python datasets/split_data/gta/split_gta.py
```
For the GTA dataset, a small set of label maps (60 frames) has a different resolution than their corresponding image.
Therefore, we need to resize these label maps.
```
python datasets/split_data/gta/resize_img.py
mv datasets/GTA/labels/valid_resize/* datasets/GTA/labels/valid/
rm -rf datasets/GTA/labels/valid_resize/
```
Finally, we map the labels for detectron2:
```
python datasets/prepare_gta_sem_seg.py
```

## Expected dataset structure for [Synthia](https://synthia-dataset.net/downloads/):
```
synthia/
    Depth/
        Depth
    GT/
        COLOR/
        LABELS/
            train/
            val/
    RGB/
        train/
        val/
```
We follow the [RobustNet](https://github.com/shachoi/RobustNet) to split the dataset.
```
python datasets/synthia/split_synthia.py
```
We then map the labels from synthia to cityscapes.
```
python datasets/prepare_synthia_sem_seg.py
```