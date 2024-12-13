# classification-challenge
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Import the data
data = pd.read_csv("https://static.bc-edx.com/ai/ail-v-1-0/m13/challenge/spam-data.csv")
data.head()

#I predict Random Forest will be the most accurate.*

# Create the labels set `y` and features DataFrame `X`
y = data['spam']

X = data.drop(columns=['spam'])
X.head()

# Check the balance of the labels variable (`y`) by using the `value_counts` function.
data["spam"].value_counts()

# Split the data into X_train, X_test, y_train, y_test
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

from sklearn.preprocessing import StandardScaler

# Create the StandardScaler instance
X_scaler = StandardScaler()

# Fit the Standard Scaler with the training data
X_scaler.fit(X_train) 

# Scale the training data
X_train_scaled = X_scaler.transform(X_train)
X_test_scaled = X_scaler.transform(X_test)

# Train a Logistic Regression model and print the model score
from sklearn.linear_model import LogisticRegression

# Make and save testing predictions with the saved logistic regression model using the test data
lr_model = LogisticRegression(random_state=1)
lr_model.fit(X_train_scaled, y_train)
testing_predictions = lr_model.predict(X_test_scaled)

# Review the predictions
print(f"Training Data Score: {lr_model.score(X_train_scaled, y_train)}")
print(f"Testing Data Score: {lr_model.score(X_test_scaled, y_test)}")

# Calculate the accuracy score by evaluating `y_test` vs. `testing_predictions`.
acc_score = accuracy_score(y_test, testing_predictions) 
acc_score

# Train a Random Forest Classifier model and print the model score
from sklearn.ensemble import RandomForestClassifier

# Make and save testing predictions with the saved logistic regression model using the test data
rf_model = RandomForestClassifier(random_state=1)
rf_model.fit(X_train_scaled, y_train)
rf_testing_predictions = rf_model.predict(X_test_scaled)


# Review the predictions
print(f"Training Data Score: {rf_model.score(X_train_scaled, y_train)}")
print(f"Testing Data Score: {rf_model.score(X_test_scaled, y_test)}")

# Calculate the accuracy score by evaluating `y_test` vs. `testing_predictions`.
y_test_acc_score = accuracy_score(y_test, testing_predictions)
y_test_acc_score

## Evaluate the Models

*Which model performed better? How does that compare to your prediction? Write down your results and thoughts in the following markdown cell.*

*Both models had an accuracy score of 0.9226759339704604. The RandomForestClassifier performed better than the Logisic Regression model. As I assumed, the RandomForestClassifier performed better. The RandomForestClassifier testing data score was 0.9582971329278888 and Logistic Regression model was lower with a testing data score of 0.9226759339704604.* 

