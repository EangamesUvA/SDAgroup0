from dataset import *
import matplotlib.pyplot as plt

# ------------------------------
# Box Plots
# ------------------------------

def box_plots(indep_vars, dep_vars):
    for indep in indep_vars:
        indep_label = MAPPING[indep]
        for dep in dep_vars:
            dep_label = MAPPING[dep]

            plt.figure(figsize=(10, 5))
            DATASET.data.boxplot(column=dep, by=indep)
            plt.title(f"{dep_label} by {indep_label}")
            plt.suptitle("")  # remove default subtitle
            plt.xlabel(indep_label)
            plt.ylabel(dep_label)
            plt.xticks(rotation=45)
            plt.tight_layout()

    plt.show()


# ------------------------------
# Histogram
# ------------------------------

def plot_trstplt_histogram():
    trstplt = DATASET.data["trstplt"]

    plt.figure()
    plt.hist(trstplt)
    plt.xlabel("Trust in politicians (trstplt)")
    plt.ylabel("Frequency")
    plt.title("Histogram of Trust in Politicians")
    plt.tight_layout()
    plt.show()


# ------------------------------
# Main
# ------------------------------

if __name__ == "__main__":
    box_plots(INDEP_VAR, DEP_VAR)
    plot_trstplt_histogram()
