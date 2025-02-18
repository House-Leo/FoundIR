We provide two specialist models, i.e., **Lowlight** and **Weather** models, to refine the results of the generalist model. 

In our experiments, we refine the generalist model's outputs as follows: 
```
Weather model: 0501-0700 and 1051-1100 inputs
Lowlight model: 0701-0800, 1101-1250, and 1301-1500 inputs 
```

**Please note that this is optional**, allowing you to further refine the generalist model’s outputs using the following commands, especially for challenging **lowlight, hazy, and rainy** inputs.

- Install the environment.
```
cd ./specialist_model
pip install -r requirements.txt
python setup.py develop
```
- Put the testset in the `./dataset` folder.
- Run the following command to test the specialist models.
```
python inference_lowlight.py
or
python inference_weather.py
```
And you can find the output visual results in the folder `results/`.
