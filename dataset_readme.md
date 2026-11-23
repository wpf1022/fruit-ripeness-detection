---
dataset_info:
  features:
  - name: image
    dtype: image
  - name: label
    dtype:
      class_label:
        names:
          '0': Raw_Banana
          '1': Raw_Mango
          '2': Ripe_Banana
          '3': Ripe_Mango
  splits:
  - name: train
    num_bytes: 279368762.236
    num_examples: 3999
  - name: test
    num_bytes: 35482482
    num_examples: 1000
  download_size: 390936312
  dataset_size: 314851244.236
configs:
- config_name: default
  data_files:
  - split: train
    path: data/train-*
  - split: test
    path: data/test-*
license: apache-2.0
task_categories:
- image-classification
language:
- en
tags:
- biology
pretty_name: fruit_ripeness_img
---
# Dataset Card for Fruit-Ripeness-Classification dataset

<!-- Provide a quick summary of the dataset. -->

This is a collection of ripe and unripe fruits (mangoes and bananas) in outside lighting and outside conditions.
- Train - 80% (4k images)
- Test - 20% (1k images)

Dimensions of image : 640 x 480

The dataset has been collected from Mendeley data: https://data.mendeley.com/datasets/y3649cmgg6/3 (Mango and Banana Dataset (Ripe Unripe) : Indian RGB image datasets for YOLO object detection)

Initially the data was for training YOLO models.
I have reorganized the data for training using datasets library in python for deep neural networks and transformers.



## Dataset Details

## Uses

This dataset is intended for image classification purpose.

## Dataset Card Authors

Subhajit Chatterjee

## Dataset Card Contact

chatterjeesubhajit027@gmail.com