# This sciprt is used for preprocessing the AI feature
from utils.parser import get_args
from data.data_process import GraphDataModule

print("# Running Preprocess data script.")
args = get_args()
dm = GraphDataModule(args)
dm.setup()
