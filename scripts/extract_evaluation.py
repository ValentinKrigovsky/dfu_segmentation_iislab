import csv
import argparse
import json
import os


def write_to_csv(headers, data, new) -> None:
    with open('/home/dfu/first_nnu-net_experiments/predictions/extracted_predictions.csv', 'a') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=headers)
        if new:
            writer.writeheader()
        writer.writerow(data)


def preprocess_data(data, task_path):
    """Need to return one dict for every line in csv. One line - one model prediction"""
    data0 = data["results"]["mean"]["0"]
    data1 = data["results"]["mean"]["1"]
    old_keys = data0.keys()
    keys0 = [key + "0" for key in data0]
    keys1 = [key + "1" for key in data0]
    headers = ["Path"] + keys0 + keys1

    return_data = {"Path": task_path}

    for new_key, old_key in zip(keys0, old_keys):
        return_data[new_key] = data0[old_key]

    for new_key, old_key in zip(keys1, old_keys):
        return_data[new_key] = data1[old_key]

    return headers, return_data


def main():
    # assign directory
    directory = '/home/dfu/first_nnu-net_experiments/predictions'

    # iterate over files in dir
    new = True
    best0 = (0, "")
    best1 = (0, "")
    for filename in os.listdir(directory):

        task_path = os.path.join(directory, filename + "/summary.json")

        if not os.path.isfile(task_path):
            continue

        json_file = open(task_path)
        #   get data from json
        data = json.load(json_file)
        headers, data = preprocess_data(data, filename)

        if data["Dice0"] > best0[0]:
            best0 = (data["Dice0"], data["Path"])
        if data["Dice1"] > best1[0]:
            best1 = (data["Dice1"], data["Path"])

        #   write data to csv
        write_to_csv(headers, data, new)
        new = False

    print(f"Best background: {best0[0]} | {best0[1]}")
    print(f"Best ulcer: {best1[0]} | {best1[1]}")


if __name__ == "__main__":
    main()
