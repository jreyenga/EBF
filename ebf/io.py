# -*- coding: utf-8 -*-
"""
Model checkpoint save and restore (TF2 — tf.train.Checkpoint + JSON sidecar).
"""
import json
import os
import tensorflow as tf


def save(model, optimizer, path, filename, scale=None, offset=None):
    """Save model weights via ``tf.train.Checkpoint`` and config as JSON sidecar.

    Parameters
    ----------
    model : EBFModel
        Trained model instance.
    optimizer : tf.keras.optimizers.Optimizer or None
        Optimizer instance (its state is saved alongside model weights).
        ``None`` saves model weights only — e.g. when re-saving a model
        that was restored via :func:`restore` and has no optimizer.
    path : str
        Directory for checkpoint files.
    filename : str
        Checkpoint filename stem.
    scale : numpy.ndarray or None, optional
        Scale array to store in the JSON sidecar.
    offset : numpy.ndarray or None, optional
        Offset array to store in the JSON sidecar.

    Returns
    -------
    ckpt_path : str
        Full checkpoint file stem (pass to :func:`restore`).
    """
    ckpt_path = os.path.join(path, filename)
    if optimizer is not None:
        ckpt = tf.train.Checkpoint(model=model, optimizer=optimizer)
    else:
        ckpt = tf.train.Checkpoint(model=model)
    ckpt.write(ckpt_path)

    # JSON sidecar: model config needed to reconstruct the Module before restore
    config = {
        'n_dims': model.n_dims,
        'n_nodes': model.n_nodes,
        'basis': model.basis,
        'eps': model.eps,
    }
    if scale is not None:
        config['Scale'] = scale.tolist()
    if offset is not None:
        config['Offset'] = offset.tolist()

    json_path = ckpt_path + '.json'
    with open(json_path, 'w') as f:
        json.dump(config, f, indent=2)

    return ckpt_path


def restore(file):
    """Restore model from checkpoint + JSON sidecar.

    Parameters
    ----------
    file : str
        Checkpoint file stem as returned by :func:`save`.

    Returns
    -------
    model : EBFModel
        Model with restored weights.
    config : dict
        Dictionary with keys ``'n_dims'``, ``'n_nodes'``, ``'basis'``,
        ``'eps'``, and optionally ``'Scale'``, ``'Offset'``.
    """
    from ebf.model import EBFModel  # deferred to avoid circular import

    json_path = file + '.json'
    with open(json_path, 'r') as f:
        config = json.load(f)

    model = EBFModel(
        n_dims=config['n_dims'],
        n_nodes=config['n_nodes'],
        basis=config['basis'],
        eps=config['eps'],
    )

    ckpt = tf.train.Checkpoint(model=model)
    ckpt.restore(file).expect_partial()

    return model, config
