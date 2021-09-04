# Tenserfelow
** This repo will guide you to train using tensorflow for custom object dtection with tensorflow 2.X version.


Workspace env:

    CUDA VERSION : 11.4
    TF VERSION: 2.6.0
    NUMPY VERSION: 1.19.5

  
**Before annotate images, make sure resize the image using

    python3 image_resizer.py

~~~
pip3 install tensorflow-gpu
~~~


directory structure should be like this :

u should add pre-trained-models, models, images, annotations.

~~~
 TensorFlow_training/
└─ models/
   ├─ community/
   ├─ official/
   ├─ orbit/
   ├─ research/
└─ scripts/
└─ workspace/
   ├─ training_demo/
   
 ~~~
 
 install :
    
    git clone https://github.com/tensorflow/models.git



cd models\research

    protoc object_detection/protos/*.proto --python_out=.
    $ pip3 install tf_slim
    $ pip3 install pandas

    pip3 install cython
    pip3 install git+https://github.com/philferriere/cocoapi.git#subdirectory=PythonAPI

cd \TensorFlow\models\research

    copy object_detection\packages\tf2\setup.py .
    paste on research folder 
    python -m pip install .
    

To check setup is working:

    python3 object_detection/builders/model_builder_tf2_test.py
    
 ```
[       OK ] ModelBuilderTF2Test.test_create_ssd_models_from_config
[ RUN      ] ModelBuilderTF2Test.test_invalid_faster_rcnn_batchnorm_update
[       OK ] ModelBuilderTF2Test.test_invalid_faster_rcnn_batchnorm_update
[ RUN      ] ModelBuilderTF2Test.test_invalid_first_stage_nms_iou_threshold
[       OK ] ModelBuilderTF2Test.test_invalid_first_stage_nms_iou_threshold
[ RUN      ] ModelBuilderTF2Test.test_invalid_model_config_proto
[       OK ] ModelBuilderTF2Test.test_invalid_model_config_proto
[ RUN      ] ModelBuilderTF2Test.test_invalid_second_stage_batch_size
[       OK ] ModelBuilderTF2Test.test_invalid_second_stage_batch_size
[ RUN      ] ModelBuilderTF2Test.test_session
[  SKIPPED ] ModelBuilderTF2Test.test_session
[ RUN      ] ModelBuilderTF2Test.test_unknown_faster_rcnn_feature_extractor
[       OK ] ModelBuilderTF2Test.test_unknown_faster_rcnn_feature_extractor
[ RUN      ] ModelBuilderTF2Test.test_unknown_meta_architecture
[       OK ] ModelBuilderTF2Test.test_unknown_meta_architecture
[ RUN      ] ModelBuilderTF2Test.test_unknown_ssd_feature_extractor
[       OK ] ModelBuilderTF2Test.test_unknown_ssd_feature_extractor
----------------------------------------------------------------------
Ran 20 tests in 45.304s

OK (skipped=1)
```
So meaning we good to proceed. LETSSS !!

### Gather and label dataset.

Now all the files that needed will be located in the workspace/training_demo directory. Inside the directory we have:

~~~
└─ training_demo/
   ├─ annotations/
   ├─ images/
   ├─ models/
   ├─ pre-trained-models/
 
 ~~~
 
After gathering some images, you must partition the dataset. seperate the data in to a training set and testing set. You should put 85% of your images in to the images\train folder and put the remaining 15% in the images\test folder. After seperating your images, you can label them with [LabelImg](https://tzutalin.github.io/labelImg).
 
### Generating Training Data

Since our images and XML files are prepared, we are ready to create the label_map. It is located in the annotations folder, so navigate to the folder. After you've located label_map.pbtxt, open it with a Text Editor of your choice. Since my model had 3 classes of ships, my labelmap looked like:

```
item {
    id: 1
    name: 'military_ship'
}

item {
    id: 2
    name: 'speedboat'
}

item {
    id: 3
    name: 'fishing_boat'
}
```

Once done. saved it as label_map.pbtxt. Now lets generate RECORD files. The code at Tensorflow/scripts/preprocessing directory.

    cd Tensorflow/scripts/preprocessing

Run:
~~~
python3 generate_tfrecord.py -x /Tensorflow/workspace/training_demo/images/train -l /Tensorflow/workspace/training_demo/annotations/label_map.pbtxt -o /Tensorflow/workspace/training_demo/annotations/train.record

python3 generate_tfrecord.py -x /Tensorflow/workspace/training_demo/images/train -l /Tensorflow/workspace/training_demo/annotations/label_map.pbtxt -o /Tensorflow/workspace/training_demo/annotations/test.record
~~~

Now under annotations directory shouldhave train.record and test.record


Download Tensorflow pre-trained model from zoo model. Download and  put inside pre-trained-models directory. I use the [SSD MobileNet V2 FPNLite 640x640](http://download.tensorflow.org/models/object_detection/tf2/20200711/ssd_mobilenet_v2_fpnlite_640x640_coco17_tpu-8.tar.gz). u can use any other model as long ada v2(version2).

The structure directory shoul be look like this.

```
training_demo/
├─ ...
├─ pre-trained-models/
│  └─ ssd_mobilenet_v2_fpnlite_640x640_coco17_tpu-8/
│     ├─ checkpoint/
│     ├─ saved_model/
│     └─ pipeline.config
└─ ...
```
Now,  create a directory to store our training pipeline. Navigate to the models directory and create folder called my_mobilenet. copy pipeline.config from the pre-trained-model to my_mobilenet directory. so it should be like this:

```
training_demo/
├─ ...
├─ models/
│  └─ my_mobile/
│     └─ pipeline.config
└─ ...
```

Make changes inside model pipeline config.open models/my_mobilenet/pipeline.config:


    Line 3. Change num_classes to the number of classes your model detects. For the basketball, baseball, and football, example you would change it to num_classes: 3
    Line 135. Change batch_size according to available memory (Higher values require more memory and vice-versa). I changed it to:
        batch_size: 6
    Line 165. Change fine_tune_checkpoint to:
        fine_tune_checkpoint: "pre-trained-models/ssd_mobilenet_v2_fpnlite_640x640_coco17_tpu-8/checkpoint/ckpt-0"
    Line 171. Change fine_tune_checkpoint_type to:
        fine_tune_checkpoint_type: "detection"
    Line 175. Change label_map_path to:
        label_map_path: "annotations/label_map.pbtxt"
    Line 177. Change input_path to:
        input_path: "annotations/train.record"
    Line 185. Change label_map_path to:
        label_map_path: "annotations/label_map.pbtxt"
    Line 189. Change input_path to:
        input_path: "annotations/test.record"


Now all good to go. back to training demo directory:

### Train your model:

    python3 model_main_tf2.py --model_dir=models\my_mobilenet --pipeline_config_path=models\my_mobilenet\pipeline.config

When you are training u willse lots of warnings but as long no errors sabo and bior. once the training happen you will see something like this.

```
INFO:tensorflow:Step 100 per-step time 0.640s loss=0.454
I0810 11:56:12.520163 11172 model_lib_v2.py:644] Step 100 per-step time 0.640s loss=0.454
```

### Exporting the inference graph

Once finished training, run this command:

```
python3 ./exporter_main_v2.py --input_type image_tensor --pipeline_config_path ./models/my_mobilenet/pipeline.config --trained_checkpoint_dir ./models/my_mobilenet/ --output_directory ./exported-models/my_mobilenet_model
```
    
evaluating model:

~~~
python3 model_main_tf2.py --pipeline_config_path models/my_mobilenet/pipeline.config --model_dir models/my_mobilenet --checkpoint_dir models/my_mobilenet --alsologtostderr
~~~
### Testing out detection

~~~
python3 TF-image-od.py --model exported-models/my_mobilenet_model --labels exported-models/my_mobilenet_model/saved_model/label_map.pbtxt --image images/test/twist_06.jpg --threshold 0.60
~~~

IF ONLY GOT PROBLEM WITH YOUR ENVIRONMENT:
    
 ERROR:
 
 NotImplementedError: Cannot convert a symbolic Tensor (strided_slice:0) to a numpy array.:
def _constant_if_small(value, shape, dtype, name):
  try:
    if np.prod(shape) < 1000:
      return constant(value, shape=shape, dtype=dtype, name=name)
  except TypeError:
    # Happens when shape is a Tensor, list with Tensor elements, etc.
    pass
  return None

    solved:
    from tensorflow.python.ops.math_ops import reduce_prod
    np.prod change to reduce_prod
 



testing docker:

get pb file: 

    python3 ./exporter_main_v2.py --input_type image_tensor --pipeline_config_path ./models/my_model/pipeline.config --trained_checkpoint_dir ./models/my_model/ --output_directory ./exported-models/my_mobilenet_model



evaluating model:

    python3 TF-image-od.py --model exported-models/my_mobilenet_model --labels exported-models/my_mobilenet_model/saved_model/label_map.pbtxt --image images/test/twist_19.jpg --threshold 0.60


<p align="center">
  <img src="doc/Object Detector_screenshot_03.09.2021.png">
</p>
