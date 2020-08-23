import pandas as pd
import random as rd

rd.seed(2020)

def read_csv(filename):
    file = pd.read_csv(filename)
    file.fillna(" ", inplace=True)

    return file


def save_csv(filename, data):
    data.to_csv(filename, mode="w", index=False)


def train_shuffle_before_after(data):

    data_size = len(data)
    data["INDEX"] = range(data_size)
    data.reset_index(inplace=True, drop=True)

    positive_data = data.copy()[["INDEX", "ORIGINAL_INDEX", "LABEL", "AFTER_HEADLINE", "AFTER_BODY", "BEFORE_HEADLINE", "BEFORE_BODY"]]

    negative_data = data.copy()
    negative_data["LABEL"] = 1

    negative_data = negative_data[["INDEX", "ORIGINAL_INDEX", "LABEL", "AFTER_HEADLINE", "AFTER_BODY", "BEFORE_HEADLINE", "BEFORE_BODY"]]
    negative_data.rename(columns={"AFTER_HEADLINE": "BEFORE_HEADLINE", "AFTER_BODY": "BEFORE_BODY", "BEFORE_HEADLINE":
        "AFTER_HEADLINE", "BEFORE_BODY": "AFTER_BODY"}, inplace=True)

    concat_data = pd.concat([negative_data, positive_data])
    shuffled_data = concat_data.sample(frac=1)
    
    return shuffled_data


def shuffle_before_after(data):

    data_size = len(data)
    data["INDEX"] = range(data_size)
    data.reset_index(inplace=True, drop=True)

    shuffle_idx = rd.sample(range(data_size), k=data_size//2)

    cond = data["INDEX"].isin(shuffle_idx)

    positive_data = data[~cond]
    positive_data = positive_data[["INDEX", "ORIGINAL_INDEX", "LABEL", "AFTER_HEADLINE", "AFTER_BODY", "BEFORE_HEADLINE", "BEFORE_BODY"]]

    negative_data = data[cond]
    negative_data["LABEL"] = 1

    negative_data = negative_data[["INDEX", "ORIGINAL_INDEX", "LABEL", "AFTER_HEADLINE", "AFTER_BODY", "BEFORE_HEADLINE", "BEFORE_BODY"]]
    negative_data.rename(columns={"AFTER_HEADLINE": "BEFORE_HEADLINE", "AFTER_BODY": "BEFORE_BODY", "BEFORE_HEADLINE":
        "AFTER_HEADLINE", "BEFORE_BODY": "AFTER_BODY"}, inplace=True)

    shuffled_data = pd.concat([negative_data, positive_data])

    return shuffled_data


def split_train_valid_test(data, ratio=(0.75, 0.20, 0.05)):

    num_data = len(data)
    print("Number of dataset : {}".format(num_data))

    data.rename(columns={"INDEX": "ORIGINAL_INDEX"}, inplace=True)
    rd_ind = rd.sample(range(num_data), num_data)
    data = data.reindex(rd_ind)
    data['LABEL'] = 0

    train_end_id = int(num_data * ratio[0])
    valid_end_id = int(num_data * ratio[1])

    train_data = data.iloc[:train_end_id]
    valid_data = data.iloc[train_end_id:train_end_id + valid_end_id]
    test_data = data.iloc[train_end_id + valid_end_id:]

    return train_data, valid_data, test_data


def main():
    DATA_PATH = "data_in/training.csv"
    TRAIN_SAVE_PATH = "data_in/train.csv"
    VALID_SAVE_PATH = "data_in/dev.csv"
    TEST_SAVE_PATH = "data_in/test.csv"

    data = read_csv(DATA_PATH)

    train_data, valid_data, test_data = split_train_valid_test(data)

    shuffled_train_data = train_shuffle_before_after(train_data)
    save_csv(TRAIN_SAVE_PATH, shuffled_train_data)

    shuffled_valid_data = shuffle_before_after(valid_data)
    save_csv(VALID_SAVE_PATH, shuffled_valid_data)

    shuffled_test_data = shuffle_before_after(test_data)
    save_csv(TEST_SAVE_PATH, shuffled_test_data)


if __name__ == "__main__":
    main()
