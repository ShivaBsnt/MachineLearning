import math
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression


def prediction_using_sklearn():
    df = pd.read_csv('test_scores.csv')
    model = LinearRegression()

    model.fit(df[['math']], df['cs'].values)
    return model.coef_, model.intercept_


def gradient_descent(x, y):
    m_curr = b_curr = 0
    n = len(x)
    learning_rate = 0.0002
    prev_cost = 0
    count = 0
    for i in range(1000000):
        count += 1
        y_predicted = m_curr * x + b_curr

        cost = (1 / n) * sum([val ** 2 for val in (y - y_predicted)])
        md = -(2 / n) * sum(x * (y - y_predicted))
        bd = -(2 / n) * sum(y - y_predicted)
        m_curr = m_curr - learning_rate * md
        b_curr = b_curr - learning_rate * bd
        if math.isclose(cost, prev_cost, rel_tol=1e-20):
            return m_curr, b_curr
        prev_cost = cost

    return m_curr, b_curr


if __name__ == "__main__":
    df = pd.read_csv('test_scores.csv')
    x = np.array(df.math)
    y = np.array(df.cs)
    m, b = gradient_descent(x, y)
    print(f"m {m}, b {b}")
    m_sklearn, b_sklearn = prediction_using_sklearn()
    print(f"m_sklearn {m_sklearn[0]}, b_sklearn {b_sklearn}")

