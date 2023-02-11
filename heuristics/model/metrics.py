from collections import Counter
import pandas as pd

from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)

import matplotlib.pyplot as plt


class Metrics:
    def __init__(self, class_mapping):
        self.class_mapping = class_mapping

    def plot_cf_matrix(self, targets, predictions):
        labels_ordered = [item[0] for item in (Counter(targets).most_common())]
        cm = confusion_matrix(
            targets, predictions, labels=labels_ordered, normalize='true'
        )
        cm_disp = ConfusionMatrixDisplay(
            confusion_matrix=cm,
            display_labels=[self.class_mapping.get(label, 'None') for label in labels_ordered],
        )
        fig, ax = plt.subplots(figsize=(15, 15))
        cm_disp.plot(ax=ax, xticks_rotation='vertical')

    def get_accuracies_df(self, targets, predictions):
        df = pd.DataFrame(
            classification_report(
                targets,
                predictions,
                labels=list(self.class_mapping.keys()),
                target_names=list(self.class_mapping.values()),
                output_dict=True,
                zero_division=0,
            )
        ).T.reset_index().sort_values('support', ascending=False)

        return df
