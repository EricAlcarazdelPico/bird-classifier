import os
import shutil
import glob

def split_data():
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
    split_valid = len(images) - split_test
    
    train_data = images[:split_train]
    valid_data = images[split_train:split_valid]
    test_data = images[split_valid:]

    for img in train_data:
      print('-->Train data')
      _move_file(dir, birds_dir, img, 'train')
      num_train += 1

    for img in valid_data:
      print('-->Valid data')
      _move_file(dir, birds_dir, img, 'valid')
      num_valid += 1

    for img in test_data:
      print('-->Test data')
      _move_file(dir, birds_dir, img, 'test')
      num_test += 1

    print(f'Total classes: {num_classes}')
    print(f'Total images: {num_imgs}')
    print(f'Total train data: {num_train}')
    print(f'Total validation data: {num_valid}')
    print(f'Total test data: {num_test}')

def _move_file(dir, birds_dir, img, data_group):
  sub_dir = os.path.join(birds_dir, data_group, dir)
  # create direcotry if not exists
  if not os.path.exists(sub_dir):
    os.makedirs(sub_dir)
  # move file to data_group directory
  shutil.copy(img, sub_dir)
  print(f'--->{sub_dir}')