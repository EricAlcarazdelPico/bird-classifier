import os
import shutil
import glob

root_dir = os.path.abspath('consolidated')

labels = []
num_classes = 0
num_imgs = 0

num_train = 0
num_valid = 0
num_test = 0

current_dir = os.getcwd()
birds_dir = os.path.join(current_dir, 'birds')
os.makedirs(birds_dir)

for dir in os.listdir(root_dir):
  labels.append(dir)
  print(f'->{dir}')
  num_classes += 1

  dir_path = os.path.join(root_dir, dir)
  images = glob.glob(dir_path +'/*.jpg')
  
  num_imgs += len(images)

  split_train = int(round(len(images)*0.7))
  split_test = int(round(len(images)*0.1))
  
  train_data = images[:split_train]
  valid_data = images[split_train:len(images) - split_test]
  test_data = images[len(images) - split_test:]

  for img in train_data:
    print('-->Train data')
    sub_dir = os.path.join(birds_dir, 'train', dir)
    # create direcotry if not exists
    if not os.path.exists(sub_dir):
      os.makedirs(sub_dir)
    # move file to train directory
    # new_dir = os.path.join(sub_dir, dir + '_' + img.split('\\\\')[-1])
    shutil.copy(img, sub_dir)
    print(f'--->{sub_dir}')
    num_train += 1

  for img in valid_data:
    print('-->Validation data')
    sub_dir = os.path.join(birds_dir, 'valid', dir)
    # create direcotry if not exists
    if not os.path.exists(sub_dir):
      os.makedirs(sub_dir)
    # move file to valid directory
    # new_dir = os.path.join(sub_dir, dir + '_' + img.split('\\\\')[-1])
    shutil.copy(img, sub_dir)
    print(f'--->{sub_dir}')
    num_valid += 1

  for img in test_data:
    print('-->Test data')
    sub_dir = os.path.join(birds_dir, 'test', dir)
    # create direcotry if not exists
    if not os.path.exists(sub_dir):
      os.makedirs(sub_dir)
    # move file to test directory
    # new_dir = os.path.join(sub_dir, dir + '_' + img.split('\\\\')[-1])
    shutil.copy(img, sub_dir)
    print(f'--->{sub_dir}')
    num_test += 1

print(f'Total classes: {num_classes}')
print(f'Total images: {num_imgs}')
print(f'Total train data: {num_train}')
print(f'Total validation data: {num_valid}')
print(f'Total test data: {num_test}')