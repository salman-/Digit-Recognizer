from Dataset import Dataset


dt1 = Dataset("../datasets/train.csv")
dt2 = Dataset("../datasets/test.csv")
dt1.get_test_train_datasets()