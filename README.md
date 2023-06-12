# factor-model
5 factor model for stocks.
Data is gathered using the Bloomberg API and requires a subsciption.
The factors are market premium, size, value, profitability, and investment.

Size premium has yet to be implemented properly. factor-score is incomplete.

## factor-score.py
Calculates factors, and a score based on those factors. The data, sorted by score, is then printed and saved to an Excel file.

## historic-factor-regression.py
Calculates factors based on annual historical data, normalizes the factors, calculates a score for each ticker/year, and performs a regression analysis using the factors and as features and the forward returns as the target variable. We use a random forest regressor to make predictions on the test set, and calculate the mean squared error (MSE) and R-squared metrics.
