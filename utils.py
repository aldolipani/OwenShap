import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tensorflow.python.keras.callbacks import EarlyStopping
from tqdm import tqdm

from models import MLP


def get_cc():
    url = "https://raw.githubusercontent.com/meauxt/credit-card-default/master/credit_cards_dataset.csv"

    df = pd.read_csv(url)

    # delete column ID
    del df['ID']

    # convert to numeric
    for column in df:
        df[column] = pd.to_numeric(df[column], errors='coerce')

    # remove records with missing values
    df.dropna(inplace=True)

    # rescale marital status (married = 0, single = 1) after dropping 'others'
    df = df[df.MARRIAGE != 3]

    # remove unknown from education
    df = df[~df.EDUCATION.isin({5, 6})]

    return df


def gen_random_mlp(size_input, hidden_layers, nodes_hidden_layers, size_output):
    conf = [size_input]
    for _ in range(np.random.choice(hidden_layers)):
        conf.append(np.random.choice(nodes_hidden_layers))
    conf.append(size_output)
    return tuple(conf)


def mc_mlp_val(x_train, y_train, size_input, hidden_layers, nodes_hidden_layers, size_output=2, runs=1000, n_samples=5):
    max_acc = 0.0
    max_conf = []

    es = EarlyStopping(monitor='val_loss', mode='min', verbose=0, patience=0, restore_best_weights=True)

    mlp_cache = {}
    for _ in tqdm(range(runs)):
        # generate random mlp
        conf = gen_random_mlp(size_input, hidden_layers, nodes_hidden_layers, size_output)

        if conf not in mlp_cache:
            acc = 0.0
            for _ in range(n_samples):
                model = MLP(conf, use_bias_input=False)

                model.compile(optimizer='adam',
                              loss='sparse_categorical_crossentropy',
                              metrics=['accuracy'])

                history = model.fit(x_train, y_train,
                                    epochs=50, batch_size=300, validation_split=0.2, callbacks=[es], verbose=0)
                acc += history.history['val_accuracy'][-1]
            acc /= n_samples

            mlp_cache[conf] = acc

            if acc > max_acc:
                max_acc = acc
                max_conf = conf

    return max_conf, max_acc


def mse(phi_hat, phi):
    return np.mean((phi_hat - phi) ** 2)


def plot_examples_in_a_row(examples, cmap='bwr', vmin=None, vmax=None):
    fig, ax = plt.subplots(1, 10, figsize=(15, 1))
    for i in range(10):
        if vmin is not None:
            ax[i].imshow(examples[i][:-1].reshape(28, 28), cmap=cmap, vmin=vmin[i], vmax=vmax[i])
        else:
            ax[i].imshow(examples[i][:-1].reshape(28, 28), cmap=cmap)
        ax[i].axis('off')
        
def variance_analysis(model, examples, shap, runs=100):
    res = []
    # for each example
    for k in tqdm(range(examples.shape[0])):
        phis = []
        stds = []
        example = examples[k, :]
        # compute prediction of the model
        y_hat = np.argmax(model.predict(example[np.newaxis]))
        for i in range(1, runs+1):
            phi = shap(example, model, y_hat, i)
            phis.append(phi)
            stds.append(np.mean(np.std(phis, axis=0)))

        res.append(stds)
        
    return np.mean(res, axis=0)

def compute_error(model, examples, cache_phi_exact, shap, runs = 200, samples = 50, tqdm_disable = False):
    res = []
    # for each example
    for k in tqdm(range(examples.shape[0]), disable=tqdm_disable):
        example = examples[k, :]
        # compute prediction of the model
        y_hat = np.argmax(model.predict(example[np.newaxis]))
        
        phi_exact = cache_phi_exact[k]
            
        error = []
        for _ in range(samples):
            phi_shap = shap(example, model, y_hat, runs)
            mse_shap = mse(phi_shap, phi_exact)
            error.append(mse_shap)

        res.append(np.mean(error))

    return res
