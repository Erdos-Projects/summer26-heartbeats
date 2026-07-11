"""
Conv1D convolutional network for cross-subject PAMAP2 activity recognition, the deep-learning
counterpart to the classical estimators in modeling.py.

Where modeling.py scores the classical estimators on the Table 3 features, this network trains on
the raw filtered windows and learns its own features, following the Conv1d-CNN of Yang et al.
(Section III-C-2 and Table 5). The function loso_cnn_scores runs the leave-one-subject-out
evaluation used in modeling.py, and cnn_cv_scores runs the same training under any cross-validation
splitter, such as GroupKFold grouped by interval. The per-channel scaling is fit on each fold's
training windows alone. The module requires keras and a backend.
"""
import os
import sys
import functools

import numpy as np
import matplotlib.pyplot as plt
import keras
from keras import models, layers
from keras.utils import to_categorical, set_random_seed
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix

# Activity names come from read_data in analysis_and_validation.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'analysis_and_validation'))
from read_data import get_activityMap

# Activity groupings, each a remap of activity_id -> group id with names for the combined groups.
# The 'raw' scheme keeps the 12 activities, 'grouped9' merges them to 9 classes, and 'lmh'
# collapses everything to low, medium, and high. A combined group reuses one of its member
# activity_ids so get_activityMap still labels it, except 'lmh', which uses 101, 102, and 103 to
# avoid colliding with real ids.
ACTIVITY_GROUPINGS = {
    'raw': {'remap': {}, 'names': {}},
    'grouped9': {
        'remap': {2: 1, 7: 4, 17: 16},
        'names': {1: 'Resting', 4: 'Walking', 16: 'House work'},
    },
    'lmh': {
        'remap': {1: 101, 2: 101, 3: 101,
                  4: 102, 7: 102, 12: 102, 13: 102, 16: 102, 17: 102,
                  5: 103, 6: 103, 24: 103},
        'names': {101: 'Low', 102: 'Medium', 103: 'High'},
    },
}


def apply_grouping(y, grouping):
    """
    Remap activity labels under a grouping. The grouping is a key of ACTIVITY_GROUPINGS such as
    'grouped9' or 'lmh', a remap dict of {old_id: new_id}, or None or 'raw' to leave the 12
    activities unchanged. Ids absent from the remap are kept as they are.
    """
    if grouping is None or grouping == 'raw':
        return np.asarray(y)
    remap = ACTIVITY_GROUPINGS[grouping]['remap'] if isinstance(grouping, str) else grouping
    return np.array([remap.get(int(v), int(v)) for v in y], dtype=int)


def grouping_names(grouping):
    """
    Return the {id: name} labels for a grouping, get_activityMap extended with the grouping's
    combined-group names.
    """
    names = dict(get_activityMap())
    if isinstance(grouping, str) and grouping in ACTIVITY_GROUPINGS:
        names.update(ACTIVITY_GROUPINGS[grouping]['names'])
    return names


def build_cnn_arrays(processor, channels=None, return_intervals=False):
    """
    Stack the windows in processor.subject_segment_dict into the (n_windows, window_length,
    n_channels) tensor a Conv1D reads, keeping each window intact instead of aggregating it to a
    feature row.

    The channels argument selects the axes to stack and defaults to the 27 motion axes
    (MOTION_AXES), leaving out heart rate as in the paper's IMU-only Conv1d-CNN. The function
    returns X as float32, y as the activity_id per window, and groups as the subject_id per window
    for the LOSO split. With return_intervals it also returns the interval_id per window, for a
    GroupKFold split grouped by interval like the classical models. Windows off the modal length
    are dropped and counted, so a short window at an interval edge cannot make the stack ragged.
    """
    if channels is None:
        channels = processor.MOTION_AXES

    # The subject comes from the dict key, not the window, so the group label holds even if a
    # window's subject_id column was never set. Empty windows are skipped.
    windows = [(int(subject), w)
               for subject, segments in processor.subject_segment_dict.items()
               for w in segments if not w.empty]

    # The modal length is measured, not assumed, because a timestamp gap could yield a short window.
    lengths = [len(w) for _, w in windows]
    target_len = max(set(lengths), key=lengths.count)

    X_list, y_list, group_list, interval_list = [], [], [], []
    dropped = 0
    for subject, window in windows:
        if len(window) != target_len:
            dropped += 1
            continue
        X_list.append(window[channels].to_numpy(dtype=np.float32))
        # One interval is one activity, so the first sample labels the whole window.
        y_list.append(int(window['activity_id'].iloc[0]))
        group_list.append(subject)
        interval_list.append(int(window['interval_id'].iloc[0]))

    if dropped:
        print(f"build_cnn_arrays: dropped {dropped} window(s) of non-modal length {target_len}")

    X = np.stack(X_list)
    y = np.array(y_list, dtype=int)
    groups = np.array(group_list, dtype=int)
    if return_intervals:
        return X, y, groups, np.array(interval_list, dtype=int)
    return X, y, groups


def activity_names(class_ids, names=None):
    """
    Map activity or group ids to names for the plot labels. The names argument overrides
    get_activityMap.
    """
    activity_map = dict(get_activityMap())
    if names:
        activity_map.update(names)
    return [activity_map.get(int(i), str(i)) for i in class_ids]


def build_cnn(input_shape, n_classes, conv_filters=(32, 64, 64), kernel_size=3,
              dense_units=64, dropout=0.5, seed=440):
    """
    Build the Conv1d-CNN of the paper (Section III-C-2, Table 5). Each entry in conv_filters adds
    one Conv1D, BatchNorm, ReLU, and MaxPool block, followed by a flatten, a dropout, a dense
    hidden layer, and a dense softmax output. The input_shape is (window_length, n_channels), for
    example (201, 27).

    The default (32, 64, 64) is three blocks, matching the paper's three convolutional layers,
    with filter counts inside the paper's range of 12 to 64. A shorter tuple builds fewer blocks.
    The kernel size is 3 or 5, the paper's choices. BatchNorm sits between the convolution and its
    ReLU, the paper's arrangement, and padding='same' leaves the time length to the pooling. The
    optimizer is Adam (Table 5). The seed makes a fold's weight initialization repeatable.
    """
    set_random_seed(seed)   # repeatable weight initialization across folds and runs

    model = models.Sequential(name="pamap2_conv1d")
    model.add(layers.Input(shape=input_shape))
    for filters in conv_filters:
        model.add(layers.Conv1D(filters, kernel_size, padding='same'))
        model.add(layers.BatchNormalization())
        model.add(layers.Activation('relu'))
        model.add(layers.MaxPooling1D(2))
    model.add(layers.Flatten())
    model.add(layers.Dropout(dropout))
    model.add(layers.Dense(dense_units, activation='relu'))
    model.add(layers.Dense(n_classes, activation='softmax'))

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model


def _standardize(X_train, X_test, eps=1e-8):
    """
    Per-channel standardization fit on the training windows alone. The mean and standard deviation
    are taken over the training windows and their time steps (axes 0 and 1) and applied to both
    splits. Fitting on the training windows only keeps the held-out subject out of the scaling. A
    near-constant channel, with standard deviation below eps, is left unscaled.
    """
    mean = X_train.mean(axis=(0, 1), keepdims=True)
    std = X_train.std(axis=(0, 1), keepdims=True)
    std = np.where(std < eps, 1.0, std)
    return (X_train - mean) / std, (X_test - mean) / std


def loso_cnn_scores(X, y, groups, *, epochs=20, batch_size=256, build_fn=None,
                    verbose=0, seed=440):
    """
    Leave-one-subject-out evaluation of the CNN, matching modeling.loso_scores. Each subject is
    held out in turn while a fresh network trains on the rest, the scaling is fit on the training
    subjects only, and the held-out subject is scored.

    The evaluation has no per-fold leakage, because the held-out subject never sets the scaling,
    selects the weights, or stops training. It is passed to fit only as validation_data, to report
    the learning curve. The one caveat is that the epoch count is fixed for every fold rather than
    tuned per subject, so picking it by reading the held-out curves would let test information leak
    back into the result. Early stopping on an inner validation subject would remove that.

    The arrays X, y, and groups come from build_cnn_arrays. The build_fn argument maps
    (input_shape, n_classes) to a compiled model and defaults to build_cnn. The return value is a
    dict holding 'subjects', the per-fold 'accuracy' and 'f1_weighted' aligned to it, 'per_subject'
    with each fold's 'history' for the curves, 'y_true' and 'y_pred' pooled across folds for one
    confusion matrix, and 'classes' giving the axis order.
    """
    build_fn = build_fn or functools.partial(build_cnn, seed=seed)

    # The activity_ids are not 0..K-1 (they run 1..7, 12, 13, ...), so encode them for to_categorical.
    classes = np.unique(y)
    n_classes = len(classes)
    encoder = LabelEncoder().fit(classes)
    y_idx = encoder.transform(y)

    input_shape = (X.shape[1], X.shape[2])
    subjects = np.unique(groups)

    per_subject = {}
    accuracy, f1_weighted = [], []
    all_true, all_pred = [], []
    for subject in subjects:
        keras.backend.clear_session()   # drop the previous fold's graph

        is_test = groups == subject
        X_train, X_test = X[~is_test], X[is_test]
        y_train, y_test = y_idx[~is_test], y_idx[is_test]
        X_train, X_test = _standardize(X_train, X_test)   # fit on the training subjects only

        model = build_fn(input_shape, n_classes)
        history = model.fit(
            X_train, to_categorical(y_train, n_classes),
            validation_data=(X_test, to_categorical(y_test, n_classes)),
            epochs=epochs, batch_size=batch_size, verbose=0,
        )

        y_pred_idx = model.predict(X_test, verbose=0).argmax(axis=1)
        acc = accuracy_score(y_test, y_pred_idx)
        f1 = f1_score(y_test, y_pred_idx, average='weighted')

        # back to activity_ids for the confusion matrix
        true_ids = encoder.inverse_transform(y_test)
        pred_ids = encoder.inverse_transform(y_pred_idx)

        per_subject[int(subject)] = {
            'accuracy': acc, 'f1_weighted': f1, 'history': history.history,
            'y_true': true_ids, 'y_pred': pred_ids,
        }
        accuracy.append(acc)
        f1_weighted.append(f1)
        all_true.append(true_ids)
        all_pred.append(pred_ids)
        if verbose:
            print(f"subject {subject}: accuracy {acc:.3f}, f1_weighted {f1:.3f}")

    return {
        'subjects': subjects,
        'accuracy': np.array(accuracy),
        'f1_weighted': np.array(f1_weighted),
        'per_subject': per_subject,
        'y_true': np.concatenate(all_true),
        'y_pred': np.concatenate(all_pred),
        'classes': classes,
    }


def cnn_cv_scores(X, y, group_labels, splitter, *, epochs=20, batch_size=256, build_fn=None,
                  verbose=0, seed=440):
    """
    Cross-validation of the CNN under an arbitrary scikit-learn splitter, grouped by group_labels.
    This generalizes loso_cnn_scores to any split. Passing GroupKFold(5) with interval ids gives
    the non-cross-subject split, and LeaveOneGroupOut() with subject ids gives the cross-subject
    (LOSO) split. A fresh network trains per fold, scaled on that fold's training rows only.

    The return value is a dict holding the per-fold 'accuracy' and 'f1_weighted', the pooled
    'y_true' and 'y_pred' as activity ids for one confusion matrix, and 'classes'.
    """
    build_fn = build_fn or functools.partial(build_cnn, seed=seed)
    classes = np.unique(y)
    n_classes = len(classes)
    encoder = LabelEncoder().fit(classes)
    y_idx = encoder.transform(y)
    input_shape = (X.shape[1], X.shape[2])

    accuracy, f1_weighted, all_true, all_pred = [], [], [], []
    for train_idx, test_idx in splitter.split(X, y_idx, group_labels):
        keras.backend.clear_session()
        X_train, X_test = _standardize(X[train_idx], X[test_idx])
        y_train, y_test = y_idx[train_idx], y_idx[test_idx]

        model = build_fn(input_shape, n_classes)
        model.fit(X_train, to_categorical(y_train, n_classes),
                  epochs=epochs, batch_size=batch_size, verbose=0)
        y_pred = model.predict(X_test, verbose=0).argmax(axis=1)

        accuracy.append(accuracy_score(y_test, y_pred))
        f1_weighted.append(f1_score(y_test, y_pred, average='weighted'))
        all_true.append(encoder.inverse_transform(y_test))
        all_pred.append(encoder.inverse_transform(y_pred))
        if verbose:
            print(f"fold {len(accuracy)}: accuracy {accuracy[-1]:.3f}, "
                  f"f1_weighted {f1_weighted[-1]:.3f}")

    return {
        'accuracy': np.array(accuracy),
        'f1_weighted': np.array(f1_weighted),
        'y_true': np.concatenate(all_true),
        'y_pred': np.concatenate(all_pred),
        'classes': classes,
    }


def plot_learning_curves(history, title="CNN"):
    """
    Plot training against held-out accuracy and loss per epoch, from one fold's history dict. The
    gap between the curves estimates the overfitting, and the held-out level is the real
    performance.
    """
    epochs = range(1, len(history['accuracy']) + 1)
    fig, (ax_acc, ax_loss) = plt.subplots(1, 2, figsize=(14, 5))

    ax_acc.scatter(epochs, history['accuracy'], label="Training")
    ax_acc.scatter(epochs, history['val_accuracy'], label="Held-out subject")
    ax_acc.set_xlabel("Epoch")
    ax_acc.set_ylabel("Accuracy")
    ax_acc.set_title(f"{title}: accuracy")
    ax_acc.legend()

    ax_loss.scatter(epochs, history['loss'], label="Training")
    ax_loss.scatter(epochs, history['val_loss'], label="Held-out subject")
    ax_loss.set_xlabel("Epoch")
    ax_loss.set_ylabel("Loss")
    ax_loss.set_title(f"{title}: loss")
    ax_loss.legend()

    fig.tight_layout()
    return fig


def plot_confusion(y_true, y_pred, classes, title="CNN", normalize=True, names=None,
                   save_path=None):
    """
    Plot the confusion matrix of the pooled held-out predictions, with rows as the true class and
    columns as the predicted class, in the imshow Blues style of the random-forest and baseline
    notebooks. When normalize is set, the default, each row is divided by its total, so the cells
    are fractions of a true class and the diagonal is per-class recall. Setting it to False gives
    raw counts. The names argument overrides the axis labels for a grouping's combined-class names,
    and save_path writes a 200-dpi PNG.
    """
    matrix = confusion_matrix(y_true, y_pred, labels=classes).astype(float)
    if normalize:
        # row-normalize, guarding the zero row of an absent class against a NaN
        row_totals = matrix.sum(axis=1, keepdims=True)
        matrix = np.divide(matrix, row_totals, out=np.zeros_like(matrix),
                           where=row_totals != 0)

    wrapped = [n.replace(' ', '\n') if len(n) > 10 else n for n in activity_names(classes, names)]

    fig, ax = plt.subplots(figsize=(10, 8))
    image = ax.imshow(matrix, cmap="Blues", vmin=0, vmax=1 if normalize else None)
    ax.set_xticks(np.arange(len(classes)))
    ax.set_yticks(np.arange(len(classes)))
    ax.set_xticklabels(wrapped, rotation=45, ha="right")
    ax.set_yticklabels(wrapped)
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    ax.set_title(f"{title} confusion matrix ({'row-normalized' if normalize else 'counts'})")

    # white text on dark cells, black on light, so the numbers stay legible
    threshold = matrix.max() / 2 if matrix.size else 0
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            value = matrix[i, j]
            text = f"{value:.2f}" if normalize else f"{int(value)}"
            ax.text(j, i, text, ha="center", va="center", fontsize=8,
                    color="white" if value > threshold else "black")

    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04,
                 label="Fraction of true class" if normalize else "Count")
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=200, bbox_inches="tight")
    return fig


def plot_model_comparison(scores_by_model, subjects, title="Leave-one-subject-out accuracy",
                          ylabel="Accuracy", save_path=None):
    """
    Plot a grouped bar chart of a per-subject score across models, with a trailing "Mean" group
    whose error bar is the across-subject standard deviation. This places the CNN's LOSO number
    beside the classical models. The scores_by_model argument is a dict of {name: per-subject
    scores}, each aligned with subjects, and save_path writes a 200-dpi PNG.
    """
    names = list(scores_by_model)
    positions = np.arange(len(subjects) + 1)   # one slot per subject, plus the mean
    width = 0.8 / len(names)

    fig, ax = plt.subplots(figsize=(12, 6))
    for k, name in enumerate(names):
        values = np.asarray(scores_by_model[name], dtype=float)
        heights = np.append(values, values.mean())
        offset = (k - (len(names) - 1) / 2) * width   # center each model's cluster in its slot
        ax.bar(positions + offset, heights, width, label=name)
        ax.errorbar(positions[-1] + offset, values.mean(), yerr=values.std(),
                    fmt="none", ecolor="black", capsize=3)   # spread, on the Mean bar only

    ax.set_xticks(positions)
    ax.set_xticklabels([f"S{int(s)}" for s in subjects] + ["Mean"])
    ax.set_xlabel("Held-out subject")
    ax.set_ylabel(ylabel)
    ax.set_ylim(0, 1)
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=200, bbox_inches="tight")
    return fig


def plot_accuracy_by_grouping(results_df, prefix="acc_", save_path=None):
    """
    Plot the mean and standard deviation of LOSO accuracy per grouping, from the acc_<scheme>
    columns of a per-subject results table (cnn_loso_results.csv). The bars show the trend as the
    labels are coarsened.
    """
    cols = [c for c in results_df.columns if c.startswith(prefix)]
    schemes = [c[len(prefix):] for c in cols]
    means = [results_df[c].mean() for c in cols]
    stds = [results_df[c].std() for c in cols]

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.bar(schemes, means, yerr=stds, capsize=4)
    for i, m in enumerate(means):
        ax.text(i, m + 0.015, f"{m:.2f}", ha="center")
    ax.set_ylim(0, 1)
    ax.set_xlabel("grouping")
    ax.set_ylabel("LOSO accuracy")
    ax.set_title("CNN accuracy by activity grouping")
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=200, bbox_inches="tight")
    return fig


def plot_score_boxplot(scores_by_model, title="Per-subject LOSO accuracy",
                       ylabel="LOSO accuracy", save_path=None):
    """
    Plot a box plot of the per-subject scores for each model, one box per model, with the
    individual subjects overlaid as points. The boxes show the spread across subjects, and the
    outliers, that a mean hides.
    """
    names = list(scores_by_model)
    data = [np.asarray(scores_by_model[n], dtype=float) for n in names]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.boxplot(data, showmeans=True)
    for i, arr in enumerate(data, start=1):
        ax.scatter(np.full(len(arr), i), arr, color="tab:gray", alpha=0.6, zorder=3)
    ax.set_xticks(range(1, len(names) + 1))
    ax.set_xticklabels(names)
    ax.set_ylim(0, 1)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=200, bbox_inches="tight")
    return fig
