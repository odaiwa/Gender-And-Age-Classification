import os
import glob
import random
import shutil

m_path = r'F:\לימודים\פרויקט גמר\project\ICDAR_gender_language\arabic\male'
f_path = r'F:\לימודים\פרויקט גמר\project\ICDAR_gender_language\arabic\female'

split_dir = r'F:\לימודים\פרויקט גמר\project\ICDAR_gender_split_language\arabic'

os.chdir(m_path)
print(os.getcwd())

for c in random.sample(glob.glob("*.jpg"), 362):
     shutil.move(c, split_dir + r'\train\male')

for c in random.sample(glob.glob("*.jpg"), 40):
     shutil.move(c, split_dir + r'\test\male')

for c in random.sample(glob.glob("*.jpg"), 40):
     shutil.move(c, split_dir + r'\valid\male')

os.chdir(f_path)
print(os.getcwd())

for c in random.sample(glob.glob("*.jpg"), 362):
     shutil.move(c, split_dir + r'\train\female')

for c in random.sample(glob.glob("*.jpg"), 40):
     shutil.move(c, split_dir + r'\test\female')

for c in random.sample(glob.glob("*.jpg"), 40):
     shutil.move(c, split_dir + r'\valid\female')