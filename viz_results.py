"""Hyperparameter heatmaps (paper Figures 3-5), written to figures/."""

import json, os
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import config, veccisc

FIGURES = {
    'T': 'heatmap_T.png',            # Figure 3: softmax temperature T
    'C': 'heatmap_C_KMeans.png',     # Figure 4: K for VecCISC + KMeans
    'C_HAC': 'heatmap_C_HAC.png',    # Figure 5: K for VecCISC + HAC
}

def viz_hyperparameters(param:str, name:str = None, hp_file:str = 'hyperparameters.json', show:bool = False) -> None:
    with open(os.path.join(config.RESULTS_DIR, hp_file)) as f:
        hp = json.load(f)

    models, ds = [*hp], [*hp[[*hp][0]]]
    ds_labels = [veccisc.DS_TO_NAME[i] for i in ds]
    model_labels = [veccisc.MODELS_TO_NAME[i] for i in models]

    t = np.array([[round(hp[model][d][param], 2) for d in ds] for model in models])

    fig, ax = plt.subplots()
    ax.imshow(t)

    ax.set_xticks(range(len(ds_labels)), labels=ds_labels,
                rotation=45, ha="right", rotation_mode="anchor")
    ax.set_yticks(range(len(model_labels)), labels=model_labels)

    for i in range(len(model_labels)):
        for j in range(len(ds_labels)):
            ax.text(j, i, t[i, j], ha="center", va="center", color="w")

    ax.set_title("")
    fig.tight_layout()
    if name is not None:
        os.makedirs(config.FIGURES_DIR, exist_ok=True)
        plt.savefig(os.path.join(config.FIGURES_DIR, name), dpi=300)

    if show:
        plt.show()

    plt.close(fig)


if __name__ == '__main__':
    matplotlib.use('Agg')
    for param, name in FIGURES.items():
        viz_hyperparameters(param, name)
        print(os.path.join(config.FIGURES_DIR, name))
