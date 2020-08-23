import numpy as np


def main():
    ensemble_logits = np.empty((99,2))

    with open("ensemble/ensemble_candidate.txt", "r") as f:
        for idx, line in enumerate(f.readlines()):
            logits = np.array(eval(line))
            ensemble_logits += logits

    ensemble_logits /= (idx + 1)
    print(idx + 1)
    print(ensemble_logits)
    prediction = np.argmax(ensemble_logits, axis=1)

    with open("ensemble/answer.csv", 'w') as f:
        for idx, pred in enumerate(prediction):
            f.writelines("{}, {}, {}\n".format(idx+1, 0 if not pred else 1, 0 if pred else 1))
    with open("ensemble/ensemble_result.txt", 'w') as f:
        f.writelines("{}\n".format(prediction.tolist()))
    print("Finsh")

    
if __name__ == "__main__":
    main()