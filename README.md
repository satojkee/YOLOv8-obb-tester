# [YOLOv8-obb](https://docs.ultralytics.com/models/yolov8/) test util

## Dependency installation
```shell
pip3 install -r requirements.txt
```


## Configuration
> Be sure that `setup.json` file is in the project root or use `--setup | -s` option to provide it

```
root/
|-- examples/
|   |-- dice_1.jpg
|   |-- dice_2.jpg
|   |-- dice_3.jpg
|   |-- planes_1.jpg
|   |-- planes_2.jpg
|   |-- planes_3.jpg
|   |-- planes_4.jpg
|   |-- planes_5.jpg
|   |-- planes_6.jpg
|-- src/
|   |-- config.py
|   |-- tester.py
|-- main.py
|-- requirements.txt
|-- README.md
|-- setup.json
|-- .gitignore

```

```json lines
{
  "predict": {
    // supports all params listed here [https://docs.ultralytics.com/usage/cfg/#predict-settings]
    "imgsz": 640,
    "augment": false,
    "save_conf": true,
    "verbose": true,
    "iou": 0.9,
    "conf": 0.5
  },
  "visual": {
    // label configuration
    "label": {
      "thickness": 2, // text thickness
      "fontScale": 0.7, // font scale 
      "color": [255, 255, 255] // font color (BGR)
    },
    // object bounding box configuration
    "bbox": {
      "thickness": 3, // line thickness
      "color": [0, 255, 0] // fill color (BGR)
    },
    // label bounding box configuration (applies a contrast bg behind the label)
    "lbox": {
      "color": [255, 0, 0], // fill color (BGR)
      "thickness": -1 // line thickness (use recommended '-1' value)
    }
  },
  "storage": {
    "model": "YOUR_MODEL.pt",
    "assets": "YOUR_ASSETS_DIR",
    "output": "YOUR_OUTPUT_DIR"
  }
}
```

## Usage
```shell
python main.py
```

or (if store that file somewhere else)

```shell
python main.py --setup="path/your_setup.json"
```

## Examples of `yolov8m-obb.pt`
![planes_1](https://i.imgur.com/wLeKSoK.jpeg)
![planes_2](https://i.imgur.com/gPuT3c7.jpeg)
![planes_3](https://i.imgur.com/wUszwfD.jpeg)


## Examples of `dice-recognition.pt`
> Note: it's a custom trained model based on `yolov8m-obb.pt` \
> <b>Oriented Bounding Boxes are inaccurate somewhere due to the small dataset used for training</b>

![dice_1](https://i.imgur.com/dKP4PzM.jpeg)
![dice_2](https://i.imgur.com/RqgVU4N.jpeg)
![dice_3](https://i.imgur.com/jSJXBTW.jpeg)