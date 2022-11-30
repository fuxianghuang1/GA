# Adversarial and Isotropic Gradient Augmentation for Image Retrieval with Text Feedback - Accepted at IEEE Transactions on Multimedia 2022
The paper can be accessed at: https://ieeexplore.ieee.org/document/9953564


If you find this code useful in your research then please cite
'''
@ARTICLE{GA_for_IRTF,
  author={Huang, Fuxiang and Zhang, Lei and Zhou, Yuhang and Gao, Xinbo},
  journal={IEEE Transactions on Multimedia}, 
  title={Adversarial and Isotropic Gradient Augmentation for Image Retrieval with Text Feedback}, 
  year={2022},
  volume={},
  number={},
  pages={1-12},
  doi={10.1109/TMM.2022.3222624}}
'''


## Abstract

Image Retrieval with Text Feedback (IRTF) is an emerging research topic where the query consists of an image and a text expressing a requested attribute modification. The goal is to retrieve the target images similar to the query text modified query image. The existing methods usually adopt feature fusion of the query image and text to match the target image. However, they ignore two crucial issues: overfitting and low diversity of training data, which make the feature fusion based IRTF task not generalizable. Conventional generation based data augmentation is an effective way to alleviate overfitting and improve diversity, but increases the volume of training data and generation model parameters, which is bound to bring huge computation costs. By rethinking the conventional data augmentation mechanism, we propose a plug-and-play Gradient Augmentation (GA) based regularization approach. Specifically, GA contains two items: 1) To alleviate model overfitting on the training set, we deduce an explicit adversarial gradient augmentation from the perspective of adversarial training, which challenges the “no free lunch” philosophy. 2) To improve the diversity of training set, we propose an implicit isotropic gradient augmentation from the perspective of gradient descent-based optimization, which achieves the goal of big gain but no pain. Besides, we introduce deep metric learning to train the model and provide theoretical insights of GA on generalization. Finally, we propose a new evaluation protocol called Weighted Harmonic Mean (WHM) to assess the model generalization. Experiments show that our GA outperforms the state-of-the-art methods by 6.2\% and 4.7\% on CSS and Fashion200k datasets, respectively, without bells and whistles.



## Requirements and Installation
* Python 3.6
* [PyTorch](http://pytorch.org/) 1.2.0
* [NumPy](http://www.numpy.org/) (1.16.4)
* [TensorBoard](https://github.com/TeamHG-Memex/tensorboard_logger)

## Description of the Code [(From TIRG)](https://github.com/google/tirg/edit/master/README.md)
The code is based on TIRG code. 


- `main.py`: driver script to run training/testing
- `datasets.py`: Dataset classes for loading images & generate training retrieval queries
- `text_model.py`: LSTM model to extract text features
- `img_text_composition_models.py`: various image text compostion models 
- `torch_function.py`: contains soft triplet loss function and feature normalization function
- `test_retrieval.py`: functions to perform retrieval test and compute recall performance

## Running the experiments 

### Download the datasets

### CSS3D dataset

Download the dataset from this [external website](https://drive.google.com/file/d/1wPqMw-HKmXUG2qTgYBiTNUnjz83hA2tY/view?usp=sharing).

Make sure the dataset include these files:
`<dataset_path>/css_toy_dataset_novel2_small.dup.npy`
`<dataset_path>/images/*.png`

#### MITStates dataset

Download the dataset via this [link](http://web.mit.edu/phillipi/Public/states_and_transformations/index.html) and save it in the ``data`` folder. Kindly take care that the dataset should have these files:

```data/mitstates/images/<adj noun>/*.jpg```


#### Fashion200k dataset

Download the dataset via this [link](https://github.com/xthan/fashion-200k) and save it in the ``data`` folder.
To ensure fair comparison, we employ the same test queries as TIRG. They can be downloaded from [here](https://storage.googleapis.com/image_retrieval_css/test_queries.txt). Kindly take care that the dataset should have these files:

```
data/fashion200k/labels/*.txt
data/fashion200k/women/<category>/<caption>/<id>/*.jpeg
data/fashion200k/test_queries.txt`
```



## Running the Code

For training and testing new models, pass the appropriate arguments. 

For instance, for training ComposeAE model on Fashion200k dataset run the following command:

```
python   main.py --dataset=fashion200k --dataset_path=../data/fashion200k/  --model=composeAE --loss=batch_based_classification --learning_rate_decay_frequency=50000 --num_iters=160000 --use_bert True --use_complete_text_query True --weight_decay=5e-5 --comment=fashion200k_composeAE 
```

## Notes:
### Running the BERT model
ComposeAE uses pretrained BERT model for encoding the text query. 
Concretely, we employ BERT-as-service and use Uncased BERT-Base which outputs a 768-dimensional feature vector for a text query. 
Detailed instructions on how to use it, can be found [here](https://github.com/hanxiao/bert-as-service).
It is important to note that before running the training of the models, BERT-as-service should already be running in the background.



The trained model is [here](https://pan.baidu.com/s/1ZR1ybjCSR6J9_cJm-Mu7Iw) (password：6s8y)




