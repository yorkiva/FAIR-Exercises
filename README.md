This repository is intended to create a series of notebooks that illustrate the FAIR principles in the context of High Energy Physics Data and AI models. The FAIR acronym stands for **F**indable, **A**ccessible, **I**nteroperable, and **R**eusable- benchmarking a set of desired qualities that ensure transparent and objective scientific data management.

## List of Notebooks:

- **01-Intro2FAIR.ipynb**: Introduces the FAIR principles and a set of metrics used to evaluate FAIRness of datasets
- **02-FAIRCheck-MNIST.ipynb**: Explores the MNIST dataset and its FAIRness using the metrics introduced in the previous notebook
- **03-FAIRCheck-CDMS.ipynb**: Explores the superCDMS dataset and its FAIRness
- **04a-CDMS-LR.ipynb**: Explores linear regression along with Ridge and Lasso Regularizations and preservation of these models
- **04b-CDMS-LR-FAIR.ipynb**: Explores how the model can be built, preserved, and reused in line with the FAIR principles
- **05a-CDMS-PCA.ipynb**: Explores Principal Component Analysis based linear regression. **Building a model scripts in a FAIR way and demonstrating its use via a notebook is left as an exercise**
- **07a-CDMS_NNRegressor.ipynb**: Explores a Neural Network based approach to solve the regression from on the CDMS dataset
- **07b-CDMS_NNRegressor-FAIR.ipynb**: Explores how the NN based model can be built, preserved, and reused in line with the FAIR principles
- **08a-CDMS_NNVAE.ipynb**: An introduction to Variation Auto Encoder and its implementation for the CDMS dataset
- **08b-CDMS_NNVAE.ipynb**: Explores the model building and preservation in a FAIR way. Ends with a series of exercise that will (a) allow the user to explore how model metadata can be used to reuse a preserved model and (b) modify current model to incorporate better flexibility for users
