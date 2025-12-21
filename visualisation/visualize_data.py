from data.dataset_helper import *
import matplotlib.pyplot as plt

# ------------------------------
# Box Plots
# ------------------------------

def plot_boxplots(independent_vars, dependent_vars):
    """
    Create box plots for each dependent variable
    grouped by each independent variable.
    """
    for indep in independent_vars:
        indep_label = MAPPING[indep]

        for dep in dependent_vars:
            dep_label = MAPPING[dep]

            DATASET.data.boxplot(column=dep, by=indep)
            plt.title(f"{dep_label} by {indep_label}")
            plt.suptitle("")  # remove pandas default title
            plt.xlabel(indep_label)
            plt.ylabel(dep_label)
            plt.xticks(rotation=45)
            plt.tight_layout()

    plt.show()


# ------------------------------
# Histogram
# ------------------------------

def plot_trstplt_histogram():
    """
    Plot histogram of trust in politicians.
    """
    trstplt = DATASET.data["trstplt"]

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

    # Box plots (recommended: categorical independents only)
    plot_boxplots(INDEP_VAR, DEP_VAR)

    # Histogram
    plot_trstplt_histogram()
