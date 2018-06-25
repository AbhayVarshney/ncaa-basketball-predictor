NCAA Matchup Predictor
======
Based off previous seasons, predict matchup given 2 teams.

## Description

Using previous year results, the program calculates the ELO score for each team. Using the ELO score, we use Logistic Regression as our model to predict who's going to win between the 2 teams (by returning a percentage). Then, I use SKLearn's Cross-Validation to find out the accuracy of the model. Once that is completed, the project sets up a connection with Twitter. If any user were to tweet at @NCAA_Predict, using the calculations, it would respond with who it thinks would win between the 2 teams.


## Quickstart

Install Dependencies

## 1. Run python script
python3 ncaa_predictor.py

```
$ python3 ncaa_predictor.py
/Users/avarshney/Library/Python/3.6/lib/python/site-packages/sklearn/cross_validation.py:41: DeprecationWarning: This module was deprecated in version 0.18 in favor of the model_selection module into which all the refactored classes and functions are moved. Also note that the interface of the new CV iterators are different from that of this module. This module will be removed in 0.20.
  "This module will be removed in 0.20.", DeprecationWarning)
Analyzing Season Data and computing rating based of ELO algorithm.
Total samples: 75418
Cross-validation: 0.695311
Fitting samples to Logistic Regression model.
Predicting matchups.
Converting results to csv.
Beginning Twitter communication...
Successfully obtained reply from: @Laker_Blood.
0.5675261508515781
['Arizona', 'SMU']
Successfully obtained reply from: @Laker_Blood.
0.5369924010408065
['Arizona', 'Duke']
...
```
  
## 2. Go to Twitter (https://twitter.com/NCAA_Predict)

If you tweet at the profile, then it will produce prediction based off how the 2 teams fared in the past years.