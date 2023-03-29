import argparse

# check: https://www.kaggle.com/code/mervetorkan/association-rules-with-python


def main(city: str):
    print("TODO")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--city', type=str, required=True)
    args = parser.parse_args()

    main(args.city)