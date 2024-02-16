from sklearn.metrics import r2_score

# Assuming test_x and test_y are your test data and labels
test_predictions = forward(params, test_x)

# Calculate metrics
mae = np.mean(np.abs(test_predictions - test_y))
mse = np.mean((test_predictions - test_y)**2)
r2 = r2_score(test_y, test_predictions)

print(f"\nMAE: {mae}")
print(f"MSE: {mse}")
print(f"R2 Score: {r2}")

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
ax1.scatter(test_y, test_x, color='blue')
ax1.set_title('Actual Values')
ax2.scatter(test_predictions, test_x, color='red')
ax2.set_title('Predicted Values')
plt.show()