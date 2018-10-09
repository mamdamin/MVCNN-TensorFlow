"""
constants for the data set.
ModelNet40 for example
"""
#RootFolder = '/Shared/CTmechanics_COPDGene/Amin/MV_CNN_views'
NUM_CLASSES = 3
NUM_VIEWS = 48
TRAIN_LOL = '/Shared/CTmechanics_COPDGene/Amin/MV_CNN_views/sets/train.txt'
VAL_LOL =   '/Shared/CTmechanics_COPDGene/Amin/MV_CNN_views/sets/validation.txt'
TEST_LOL =  '/Shared/CTmechanics_COPDGene/Amin/MV_CNN_views/sets/test.txt'


"""
constants for both training and testing
"""

BATCH_SIZE = 12

# this must be more than twice the BATCH_SIZE
INPUT_QUEUE_SIZE = 64 * BATCH_SIZE


"""
constants for training the model
"""
INIT_LEARNING_RATE = 0.0001

# sample how many shapes for validation
# this affects the validation time

VAL_SAMPLE_SIZE = 128

# do a validation every VAL_PERIOD iterations
VAL_PERIOD = 100

# save the progress to checkpoint file every SAVE_PERIOD iterations
# this takes tens of seconds. Don't set it smaller than 100.

SAVE_PERIOD = 2000
