"""
constants for the data set.
ModelNet40 for example
"""
<<<<<<< HEAD
NUM_CLASSES = 40
NUM_VIEWS = 12
TRAIN_LOL = './data/view/train_lists.txt'
VAL_LOL = './data/view/val_lists.txt'
TEST_LOL = './data/view/test_lists.txt'
=======
#RootFolder = '/Shared/CTmechanics_COPDGene/Amin/MV_CNN_views'
NUM_CLASSES = 3
NUM_VIEWS = 48
TRAIN_LOL = '/Shared/CTmechanics_COPDGene/Amin/MV_CNN_views/sets/train.txt'
VAL_LOL =   '/Shared/CTmechanics_COPDGene/Amin/MV_CNN_views/sets/validation.txt'
TEST_LOL =  '/Shared/CTmechanics_COPDGene/Amin/MV_CNN_views/sets/test.txt'
>>>>>>> Initial. works fine


"""
constants for both training and testing
"""
<<<<<<< HEAD
BATCH_SIZE = 16

# this must be more than twice the BATCH_SIZE
INPUT_QUEUE_SIZE = 4 * BATCH_SIZE
=======
BATCH_SIZE = 12

# this must be more than twice the BATCH_SIZE
INPUT_QUEUE_SIZE = 64 * BATCH_SIZE
>>>>>>> Initial. works fine


"""
constants for training the model
"""
INIT_LEARNING_RATE = 0.0001

# sample how many shapes for validation
# this affects the validation time
<<<<<<< HEAD
VAL_SAMPLE_SIZE = 256
=======
VAL_SAMPLE_SIZE = 128
>>>>>>> Initial. works fine

# do a validation every VAL_PERIOD iterations
VAL_PERIOD = 100

# save the progress to checkpoint file every SAVE_PERIOD iterations
# this takes tens of seconds. Don't set it smaller than 100.
<<<<<<< HEAD
SAVE_PERIOD = 1000
=======
SAVE_PERIOD = 2000
>>>>>>> Initial. works fine

