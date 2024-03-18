# Details of experimental setup
The text provides a detailed description of the experimental setup. The settings can be found in [kuairec.yaml](https://github.com/tatsuki1107/Relevance-FactorizationMachine/blob/master/conf/setting/kuairec.yaml) and [coat.yaml](https://github.com/tatsuki1107/Relevance-FactorizationMachine/blob/master/conf/setting/coat.yaml)

## 1. Details of Kuairec settings

### 1.1 Settings for generating semi-artificial datasets (`data_logging_settings`)

- **`data_path`**: KuaiRec dataset storage location
- **`train_val_test_ratio`**: Split ratio of dataset into training, validation, and testing
- **`density`**: Density of the evaluation value matrix
- **`behavior_policy`**: Algorithm for Generating Log Data. This study assumes exposure bias only and utilizes a random policy.
- **`exposure_bias`**: Strength of exposure bias

### 1.2 Table setup (`tables`)

#### 1.2.1 Interaction table

- **`data_path`**: Dataset storage location
- **`used_features`**: List of features to be used in the experiment
  - **key**: Column name
  - **value**: Data types for input to the model
  
#### 1.2.2 User table

- **`data_path`**: User feature storage location
- **`used_features`**: List of features to be used in the experiment
  - **key**: Column name
  - **value**: Data types for input to the model

#### 1.2.3 Video table

#### Daily table

- **`data_path`**: Daily video feature storage location
- **`used_features`**: List of features to be used in the experiment
  - **key**: Column name
  - **value**: Data types for input to the model

#### Category table

- **`data_path`**: Video category storage location
- **`used_features`**: List of features to be used in the experiment
  - **key**: Column name
  - **value**: Data types for input to the model

## 2. Details of Coat settings

### 2.1 Settings for spliting datasets (`data_logging_settings`)
- **`val_ratio`**: Ratio of validation data

### 2.2 Table setup (`tables`)
#### 2.2.1 Train table
- **`data_path`**: Train data storage location

#### 2.2.2 Test table
- **`data_path`**: Test data storage location

#### 2.2.3 Propensities table
- **`data_path`**: Estimated Propensities data storage location

#### 2.2.4 User features table
- **`data_path`**: User feature storage location
- **`txt_path`**: User feature columns storage location

#### 2.2.5 Item features table
- **`data_path`**: Item feature storage location
- **`txt_path`**: User feature columns storage location

## 3. Settings common to Kuairec and Coat
- **`name`**: dataset name
- **`seed`**: Seed value of random number
- **`pow_used`**: Parameters controlling the trade-off between variance and bias of the IPS estimator
- **`is_search_params`**: Flag whether or not to search for parameters before conducting the experiment. If `is_search_params=True`, only search  for the number of epochs per model and estimator.

### 3.1 Model hyperparameter settings
- **`n_factors`**: Number of dimensions of the factor
- **`reg`**: Regularization parameter
- **`batch_size`**: Batch size
- **`lr`**: Learning rate.
