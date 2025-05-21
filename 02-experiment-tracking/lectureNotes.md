# Experiment Tracking
The job of a data scientist involves a lot of experimenting in order to find the model that works best for a given problem. In the course of these experiments, there are multiple runs, and there is a need for methodological and automatic tracking of the process. Therefore, experiment tracking is the process of keeping track of all relevant information from a machine learning experiment, which includes:
- Source code
- Environments
- Data
- Model
- Parameters 
- Hyperparameters
- Metrics
- e.t.c

It is essential to track experiments for:
- Reproducability
- Organization
- Optimization

MLflow is a Python library that is used for experiment tracking, and it contains the following modules:
- Tracking
- Models
- Model Registry
- Projects

### Model Management
The machine learning lifecycle involves multiple steps to build and maintain a machine learning model.
Starting with having appropriate data, the needed steps include:
- Data sourcing
- Data labelling
- Data versioning
After the appropriate data is available, modelling begins, and this entails model management, which includes:
- Experiment tracking (model architecture, training, and evaluation)
- Model versioning
- Model deployment
- Hardware scaling
After the model is deployed, prediction monitoring is needed.

We can use MLflow to manage the model. 
