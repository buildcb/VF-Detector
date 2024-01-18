# VF-Detector - Making Multi-Granularity Code Changes on Vulnerability Fix Detector Robust to Mislabeled Changes

## Introduction
VF-Detector is a transformer-based novel techinique for detecting vulnerability-fixing commits. VF-Detector extract information of commit in respect to multiple levels of granularity (i.e. commit level, file level, hunk level, line level)

VF-Detector consists of seven feature extractors, regard the combination of granularity and CodeBERT representation:

| Feature extractor index | Granularity | CodeBERT representation |
|------------------|-------------|-------------------------|
| 1                | Commit      | Context-dependant       |
| 2                | File        | Context-dependant       |
| 3                | Hunk        | Context-dependant       |
| 5                | Commit      | Context-free            |
| 6                | File        | Context-free            |
| 7                | Hunk        | Context-free            |
| 8                | Line        | Context-free            |


To replicate the training process of VF-Detector, please follow the below steps:

        1. Finetune CodeBERT for each feature extractor
        2. Save commit embedding vectors represented by CodeBERT
        3. Train feature extractors
        4. Get confindence matrix
        5. Clean dataset
        6. Repeat Steps 1 to 3
        7. Infer feature extractors to extract commit's features
        8. Train neural classifier
        9. Apply adjustment function 
       10. Evaluate VF-Detector 

## Prerequisites
Make sure you create a directory to store embedding vectors, a folder "model" to store saved model, and a "features" folder to store extractor features following this hierarchy,and you need to put the files from the "VCC", "CLD", and "data" folder into the VF-Detector folder:
```
    VF-Detector
        model
        features
        ...
    finetuned_embeddings
        variant_1
        variant_2
        variant_3
        variant_5
        variant_6
        variant_7
        variant_8
```

Note: If you run VF-Detector on a Docker container, please run docker with parameters: "LANG=C.UTF-8 -e LC_ALL=C.UTF-8" to avoid error when writing to file, "--shm-size 16G" to avoid memory problem, "--gpus all" in case you use multiple GPUs

## Dataset
The dataset is available at: https://zenodo.org/record/5565182#.Yv3lHuxBxO8 
Please download and put dataset inside the VF-Detector folder


## Replication

Note: The current code base requires two GPUs to run. We will try to make it more flexible. 

#### Finetune CodeBERT
You need to download CodeBERT and put it in folder microsoft.
Corresponding to seven feature extractors, we have seven python scripts to finetune them.

| Feature extractor index | Finetuning script                     |
|------------------|---------------------------------------|
| 1                | python variant_1_finetune.py          |
| 2                | python variant_2_finetune.py          |
| 3                | python variant_3_finetune_separate.py |
| 5                | python variant_5_finetune.py          |
| 6                | python variant_6_finetune.py          |
| 7                | python variant_7_finetune_separate.py |
| 8                | python variant_8_finetune_separate.py |

#### Saving embedding vectors
After finetuning, run the following scripts to save embedding vectors corresponding to each feature extractor:

| Feature extractor index | Saving embeddings script                 |
|------------------|------------------------------------------|
| 1                | python preprocess_finetuned_variant_1.py |
| 2                | python preprocess_finetuned_variant_2.py |                    
| 3                | python preprocess_finetuned_variant_3.py |        
| 5                | python preprocess_finetuned_variant_5.py |           
| 6                | python preprocess_finetuned_variant_6.py |           
| 7                | python preprocess_finetuned_variant_7.py |  
| 8                | python preprocess_finetuned_variant_8.py |  

#### Saving embedding vectors 
Next, we need to train seven feature extractors

| Feature extractor index | Extractor training script                 |
|------------------|------------------------------------------|
| 1                | python variant_1.py |
| 2                | python variant_2.py |                    
| 3                | python variant_3.py |        
| 5                | python variant_5.py |           
| 6                | python variant_6.py |           
| 7                | python variant_7.py |  
| 8                | python variant_8.py |  


#### Get confindence matrix

```python3 feature_extractor_infer.py```

```python3 getPsx.py --model_path model/patch_ensemble_model.sav --java_result_path probs/prob_ensemble_classifier_test_java.txt --python_result_path probs/prob_ensemble_classifier_test_python.txt```

```python Psx.py```


#### Clean dataset

```python get_label_url.py```

```python CL.py```

```python getResult.py```


#### Infer feature extractors and train neural classifier

Simply use the following two commands:

```python3 feature_extractor_infer.py```

```python3 ensemble_classifier.py --model_path model/patch_ensemble_model.sav --java_result_path probs/prob_ensemble_classifier_test_java.txt --python_result_path probs/prob_ensemble_classifier_test_python.txt```


#### Apply adjustment function

Simply run:

```python adjustment_runner.py```

#### Evaluate VF-Detector

The script for evaluation is placed in evaluator.py

Run evaluator.py with parameter "--rq <rq_number>" to evaluate VF-Detector with the corresponding research questions:

**Performance Compared to Baselines**

```python evaluator.py --rq 1```

**Performance Affect by Training Set Size**

```python getSmallDataSet.py```

Then replicate the training process of VF-Detector

```python evaluator.py --rq 1```

**Noise Reduction Impact on Performance**

Simply modify threshold in CL.py

Then replicate the training process of VF-Detector

```python evaluator.py --rq 1```

**Performance of Effort-Aware Adjustment**

```python evaluator.py --rq 2```

**Performance of Different Levels Granularity**

To obtain performance of VF-Detector using only line level, run:

```python evaluator.py --rq 3 --mode 1```

To obtain performance of VF-Detector using only hunk level, run two commands:

```python ensemble_classifier.py --ablation_study True -v 1 -v 2 -v 5 -v 6 -v 8 --model_path model/test_ablation_line_hunk_model.sav --java_result_path probs/test_ablation_line_hunk_java.txt --python_result_path probs/test_ablation_line_hunk_python.txt```

```python evaluator.py --rq 3 --mode 2```

To obtain performance of VF-Detector using only file level, run two commands:

```python ensemble_classifier.py --ablation_study True -v 1 -v 3 -v 5 -v 7 -v 8 --model_path model/test_ablation_line_hunk_model.sav --java_result_path probs/test_ablation_line_hunk_java.txt --python_result_path probs/test_ablation_line_hunk_python.txt```

```python evaluator.py --rq 3 --mode 2```

To obtain performance of VF-Detector using only commit level, run two commands:

```python ensemble_classifier.py --ablation_study True -v 2 -v 3 -v 6 -v 7 -v 8 --model_path model/test_ablation_line_hunk_model.sav --java_result_path probs/test_ablation_line_hunk_java.txt --python_result_path probs/test_ablation_line_hunk_python.txt```

```python evaluator.py --rq 3 --mode 2```

To obtain performance of VF-Detector using only line level + hunk level, run two commands:

```python ensemble_classifier.py --ablation_study True -v 1 -v 2 -v 5 -v6  --model_path model/test_ablation_line_hunk_model.sav --java_result_path probs/test_ablation_line_hunk_java.txt --python_result_path probs/test_ablation_line_hunk_python.txt```

```python evaluator.py --rq 3 --mode 2```

To obtain performance of VF-Detector using only line level + hunk level + file level, run two commands:

```python ensemble_classifier.py --ablation_study True -v 1 -v 5  --model_path model/test_ablation_line_hunk_file_model.sav --java_result_path probs/test_ablation_line_hunk_file_java.txt --python_result_path probs/test_ablation_line_hunk_file_python.txt```

```python evaluator.py --rq 3 --mode 3```

To obtain performance of VF-Detector using all granularities.

```python evaluator.py --rq 1```
