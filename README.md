# predi_comb
Machine learning predictor-combination evaluation

When predicting a variable with machine learning algorithms, it might be of interest to learn, which combination of predictors performs best and how much the problem benefits from additional predictors.
One might also want to choose between different criteria to define ‘best performance’.

This repository holds a number of R functions to evaluate the best combination of predictors for Linear Discriminant Analysis (LDA) and Random Forest.
Since the latter has a random component, additional functions repeat the process n times, summarize the  occurring ‘best performances’ and average the respective results.

The Random Forest-related functions apply the randomForest package (Liaw and Wiener, 2002), which is an R interface to the [Fortran programs by Breiman and Cutler](https://www.stat.berkeley.edu/users/breiman/RandomForests/).
The LDA-related function applies the `lda()`-function from the MASS package (Venables and Ripley, 2002).

Each function provides a basic description as well as a detailed explanation of input variables and output at the top of the respective code.
The basic descriptions are also listed below:

## Contents of mach_learn_eval

#### randomForest.cla.eval() & randomForest.reg.eval()
These functions apply a random Forest model for classification (`...cla.eval`) and regression (`...reg.eval`) respectively and evaluate the contribution of each individual predictor as well as their combination.
Predictors are added according to their contribution measured as minimized error (specified by `criterion`).
The optimal `mtry` (number of random predictors selected for each decision node) is derived and applied for each combination, also by minimizing the specified error.

Options as optimization criterion for classifications are:
- 'OBB' (Out-of-Bag error, representing the overall error),
- each factor of the predicted variable, referring to the class error of this factor
- 'mean' (average class error)
- 'max' (maximum class error, resulting in an optimization of the worst-performing class)

Options for regression are:
- 'rmse' (root mean squared error),
- 'rsq' (pseudo R-squared: 1 - mse / Var(y))

#### randomForest.cla.avg() & randomForest.reg.avg()
These functions run the respective `randomForest.*.eval()` function n times with random (if not preset) seeds.
They compare the optimization path of predictors (order of their contribution) between the runs and average all error rates for runs with the same path.

#### lda.eval()
This function applies a Linear discriminant analysis (LDA) and evaluates the contribution of each individual predictor as well as their combination.
Predictors are added according to their contribution measured as minimized error (specified by `criterion`).

#### prep.input()
This function is called by the `*eval` functions. It takes the formula and data input handed to the parent function, applies some checks and prepares them in the format expected by the parent function: a list with the target variable as vector and all predictors in a named data.frame.

## Example
The [example.R](/example.R)-script holds a ready-to-go application of the functions in [mach_learn_eval.R](/mach_learn_eval.R).
An [example dataset on heart-disease](http://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data) is used, provided by the Center for Machine Learning and Intelligent Systems.
The dataset are prepared after [a demo script by StatQuest](https://github.com/StatQuest/random_forest_demo/blob/master/random_forest_demo.R).

## References

Liaw, A. and Wiener, M.: Classification and Regression by randomForest, R News, 2/3, 18–22, 2002.

Venables, W. N. and Ripley, B. D.: Random and Mixed Effects, in: Modern Applied Statistics with S, edited by Venables, W. N. and Ripley, B. D., Springer eBook Collection Mathematics and Statistics, pp. 271–300, Springer, New York, NY, https://doi.org/10.1007/978-0-387-21706-2_10, 2002.
