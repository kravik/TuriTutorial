import turicreate as tc

# http://vision.stanford.edu/aditya86/ImageNetDogs/
# Load images (Note: you can ignore 'Not a JPEG file' errors)
data = tc.image_analysis.load_images('stanford_dataset_small', with_path=True)
data['label'] = data['path'].apply(lambda path: path.split('/')[-2].split('-')[-1])
# Save the data for future use
data.save('dogs_small.sframe')
# Explore interactively
#data.explore()

