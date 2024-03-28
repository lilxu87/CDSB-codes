## Using Schrodinger Bridge for Model-based Offline RL

* train_stochastic.py can be viewed as Main.py

* train_eval2, train_schrodinger_bridge.py are decrepited
* the environment is in requirements.txt, you also have to download and set up mujoco
* you can directly debug train_stochastic to see how the experiment run, you don't need any  inputs from command line while debugging
* the whole learned-model part is done by conditional schrodinger bridge implemented in [CSBI github]([MSML/papers/Conditional_Schrodinger_Bridge_Imputation/configs/default_pm25_config.py at main · morganstanley/MSML (github.com)](https://github.com/morganstanley/MSML/blob/main/papers/Conditional_Schrodinger_Bridge_Imputation/configs/default_pm25_config.py)) from paper [Provably Convergent Schrödinger Bridge with Applications to Probabilistic Time Series Imputation (arxiv.org)](https://arxiv.org/pdf/2305.07247.pdf)