# Face Detection
This repo is implemented to detect human faces

## Data preparation
- Using the WIDER FACE dataset. This contains 32 203  images with 393 703  face bounding boxes
- You can download training images [here](https://drive.google.com/file/d/0B6eKvaijfFUDQUUwd21EckhUbWs/view) and bounding boxes labels [here](https://drive.google.com/file/d/1-s4QCu_v76yNwR-yXMfGqMGgHQ30WxV2/view)
- Extract and copy to the folder ```./dataset```, like this:
<pre>
Face_detection
└── dataset
    ├── images/
    ├── annotations/
</pre>
- file ```img_list.txt``` contains all paths to training images and corresponding bounding labels.
## Model
- We use the FaceBoxes to train this task. You can read the paper of FaceBoxes [here](). The model is defined in ```./models/faceboxes.py```
## Training
- To train the model, run ```python train.py```
- The checkpoints will saved in folder ```./weights``` after every 10 epoches.
## Testing
- Modify the path of testing image you want in ```test.py```. Then, run ```python test.py``` and see the results

