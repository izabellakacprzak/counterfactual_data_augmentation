import matplotlib.pyplot as plt
import numpy as np
import sigfig

def bias_estimate(original_preds, augmented_preds):
    positive_change = 0
    negative_change = 0
    for idx in range(len(original_preds)):
        original = original_preds[idx]
        augmented = augmented_preds[idx]
        if original < 0.5 and augmented > 0.5:
            positive_change += 1
        elif original > 0.5 and augmented < 0.5:
            negative_change += 1

    return (positive_change - negative_change) / (len(original_preds))


titles = ["BASELINE", "OVERSAMPLING", "STANDARD AUGMENTATIONS", "GROUP DRO", "COUNTERFACTUAL AUGMENTATIONS", "CF REGULARISATION"]
run_names = ["BASELINE", "OVERSAMPLING", "AUGMENTATIONS", "GROUP_DRO", "COUNTERFACTUALS", "CFREGULARISATION"]
for idx, run_name in enumerate(run_names):
    original_probs = []
    perturbed_probs = []
    run_name = "{}_COLORED_MNIST".format(run_name)
    with open('data/colored_mnist/originals{}.txt'.format(run_name), 'r') as fp:
        for item in fp:
            # items = item.split(', ')
            # prob = float(items[0][1:])
            item = item.strip()
            prob = float(sigfig.round(item, decimals=4))
            # prob = round(prob, 5)
            # prob = precision_round(prob, 5)
            original_probs.append(prob)
    with open('data/colored_mnist/cfs{}.txt'.format(run_name), 'r') as fp:
        for item in fp:
            # items = item.split(', ')
            # prob = float(items[0][1:])
            item = item.strip()
            prob = float(sigfig.round(str(item), decimals=4))
            # prob = round(prob, 5)
            # prob = precision_round(prob, 5)
            perturbed_probs.append(prob)


    print(len(original_probs))
    print(len(perturbed_probs))
    bias = bias_estimate(original_probs, perturbed_probs)
    print("Bias for {} is: {}".format(titles[idx], bias))
    fig = plt.figure(run_name)
    plt.scatter(np.array(original_probs), np.array(perturbed_probs), color='r', s=0.5)
    plt.axhline(y=0.5, color='black', linestyle=':')
    plt.axvline(x=0.5, color='black', linestyle=':')
    plt.title(titles[idx], y=-0.15)
    x = np.linspace(0, 1, 1000)
    plt.plot(x, x, linestyle=':', color='black')
    plt.tight_layout()
    plt.savefig("plots/fairness/colored_mnist/fairness_{}.png".format(run_name))