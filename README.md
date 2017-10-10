# EDA_using_R
Iterative factor level reduction using Decision Tree variable importance.
Reduces factor levels of categorical variables and limits the count of dummy variables.

Technique:
1. Run decision tree model (or any other model that outputs variable importance) repeatedly
2. Merge minority variable's factors into a common class at each iteration
3. Continue steps 1 and 2 as long as model accuracy does not decrease drastically (say, >20%)
4. Ultimately we obtain a simpler categorical variable (much lesser dummy variables) for further modeling
