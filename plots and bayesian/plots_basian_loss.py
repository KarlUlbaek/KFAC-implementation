import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
plt.style.use('ggplot')

matplotlib.rcParams.update({'font.size': 8})
first_params = ["basian_params/1.order_auto_v1.csv", "basian_params/1.order_class.csv", "basian_params/1.order_reg.csv"]
kfac_params = ["basian_params/kfac_auto_v1.csv", "basian_params/kfac_class_v1.csv", "basian_params/kfac_reg_v1.csv"]

names = ["auto-encoder", "classification", "regression"]
for first_df, kfac_df, name in zip(first_params, kfac_params, names):
    first_df = pd.read_csv(first_df)
    kfac_df = pd.read_csv(kfac_df)
    min_ = min(first_df.loss.values.shape, kfac_df.loss.values.shape)[0]

    plt.figure()
    plt.plot(np.flip(first_df.loss.values[:min_]), color = "indianred", label="SGD", linestyle="dotted")
    plt.plot(np.flip(kfac_df.loss.values[:min_]), color = "cornflowerblue", label="KFAC", linestyle="dotted")
    plt.title("Sorted final loss of bayesian optimization for {} training". format(name))
    plt.xlabel("Sorted bayesian iteration")
    plt.ylabel("Final Loss")
    if name == "auto-encoder":
        plt.ylim(-0.005, 0.1)
    if name == "regression":
        plt.ylim(-0.005, 0.15)
    #plt.show()
    plt.savefig("plots/bayesian_loss_{}.png".format(name), dpi=200, bbox_inches='tight')

