import random

NUM_FRAMES = 642
num_frames_train = round(NUM_FRAMES * 0.6)
num_frames_val = round(NUM_FRAMES * 0.3)
num_frames_test = NUM_FRAMES - num_frames_train - num_frames_val

# generate all indices
a = []
for i in range(NUM_FRAMES):
    a.append('%06d' % i)

# generate train indices (60%)
random.shuffle(a)
a_train = a[0:num_frames_train]
a_train.sort()
ftrain = open("train.txt", 'w+')
for i in range(len(a_train)):
    ftrain.write(a_train[i] + '\n')
ftrain.close()

# generate val indices (30%)
a_val = a[num_frames_train:num_frames_train + num_frames_val]
a_val.sort()
fval = open("val.txt", 'w+')
for i in range(len(a_val)):
    fval.write(a_val[i] + '\n')
fval.close()

# generate trainval indices
a_trainval = a_train + a_val
a_trainval.sort()
ftrainval = open("trainval.txt", 'w+')
for i in range(len(a_trainval)):
    ftrainval.write(a_trainval[i] + '\n')
ftrainval.close()

# generate test indices (10%)
a_test = a[num_frames_train + num_frames_val:]
a_test.sort()
ftest = open("test.txt", 'w+')
for i in range(len(a_test)):
    ftest.write(a_test[i] + '\n')
ftest.close()
