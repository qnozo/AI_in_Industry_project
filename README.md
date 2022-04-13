# Improve usage of unsupervised data for the definition of RUL-based maintenance policies
This is the final project for the Artificial Intelligence in Industry course at UniBo.

In the recent years, industries such as aeronautical, railway, and petroleum has transitioned from corrective or preventive maintenance to condition based maintenance (CBM). One of the enablers of CBM is Prognostics which primarily deals with prediction of remaining useful life (RUL) of an engineering asset. Besides physics-based approaches, data driven methods are widely used for prognostics purposes, however the latter technique requires availability of run to failure datasets. In this project we show a method to inject external knowledge into the deep learning model to exploit data about the normal functioning of the machines. We apply our approach on the Commercial Modular Aero-Propulsion System Simulation (CMAPSS) model developed at NASA.

## Project Work Flow

### Dataset

Commercial Modular Aero-Propulsion System Simulation (C-MAPSS), which was developedby NASA. The CMAPSS dataset includes 4 sub-datasets that are composed of multi-variate temporal data obtained from 21 sensors. Each sub-dataset contains one training set and one test set. The training datasets include run-to-failure sensor records of multiple aero-engines collected under different operational conditions and fault modes. Each engine unit starts with different degrees of initial wear and manufacturing variation that is unknown and considered to be healthy. As time progresses, the engine units begin to degrade until they reach the system failures, i.e. the last data entry corresponds to the time cycle that the engine unit is declared unhealthy. On the other hand, the sensor records in the testing datasets terminate at some
time before system failure, and the goal of this task is to estimate the remaining useful life of each engine in the test dataset. For verification, the actual RUL value for the testing engine units are also provided.
