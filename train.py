import turicreate as tc

MODEL_NAME = 'dogs'


# Load the data
data =  tc.SFrame(MODEL_NAME + '.sframe')

# Make a train-test split
train_data, test_data = data.random_split(0.8)

print('Start tc.image_classifier.create')
# Automatically pick the right model based on your data.
# Note: Because the dataset is large, model creation may take hours.
model = tc.image_classifier.create(train_data, target='label')

print('Start model.predict(test_data)')
# Save predictions to an SArray
predictions = model.predict(test_data)

print('Start model.evaluate(test_data)')
# Evaluate the model and save the results into a dictionary
metrics = model.evaluate(test_data)
print(metrics['accuracy'])

print('Saving model')
# Save the model for later use in Turi Create
model.save(MODEL_NAME + '.model')

print('Exporting to mlmodel')
# Export for use in Core ML
model.export_coreml(MODEL_NAME + '.mlmodel')
