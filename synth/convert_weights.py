#!/usr/bin/env python3
"""Convert TensorFlow checkpoint to PyTorch state dict.

Run once to generate model.pt, after which TensorFlow is no longer needed.

Usage:
    python synth/convert_weights.py
"""

import os
import numpy as np
import torch
import tensorflow.compat.v1 as tf


def reorder_lstm_gates(weights, hidden_size):
    """Reorder LSTM gates from TF [i, j, f, o] to PyTorch [i, f, g, o].

    TF gate order:  input, cell_candidate(j), forget, output
    PT gate order:  input, forget, cell_gate(g), output
    Mapping: TF[0]→PT[0], TF[1]→PT[2], TF[2]→PT[1], TF[3]→PT[3]
    """
    # Split into 4 gates along the output dimension
    i, j, f, o = np.split(weights, 4, axis=-1)
    return np.concatenate([i, f, j, o], axis=-1)


def convert_lstm_cell(reader, tf_scope, input_size, hidden_size, forget_bias=1.0):
    """Convert a TF LSTMCell to PyTorch LSTMCell weights.

    TF kernel: [input_size + hidden_size, 4 * hidden_size] (combined ih+hh)
    PT: weight_ih [4*hidden, input], weight_hh [4*hidden, hidden],
        bias_ih [4*hidden], bias_hh [4*hidden]
    """
    kernel = reader.get_tensor(f'{tf_scope}/kernel')  # [in+hid, 4*hid]
    bias = reader.get_tensor(f'{tf_scope}/bias')      # [4*hid]

    # Split kernel into input and hidden parts
    w_ih = kernel[:input_size, :]   # [input_size, 4*hidden]
    w_hh = kernel[input_size:, :]   # [hidden_size, 4*hidden]

    # Reorder gates: TF [i,j,f,o] → PT [i,f,g,o]
    w_ih = reorder_lstm_gates(w_ih, hidden_size)
    w_hh = reorder_lstm_gates(w_hh, hidden_size)
    bias = reorder_lstm_gates(bias.reshape(1, -1), hidden_size).squeeze()

    # Bake in forget_bias (TF adds forget_bias=1.0 to forget gate at runtime)
    # After reordering, forget gate is at index 1 (positions hidden_size:2*hidden_size)
    bias[hidden_size:2*hidden_size] += forget_bias

    # Transpose: TF [in, out] → PT [out, in]
    w_ih = w_ih.T
    w_hh = w_hh.T

    return {
        'weight_ih': torch.from_numpy(w_ih.copy()).float(),
        'weight_hh': torch.from_numpy(w_hh.copy()).float(),
        'bias_ih': torch.from_numpy(bias.copy()).float(),
        'bias_hh': torch.zeros(4 * hidden_size, dtype=torch.float32),
    }


def convert_linear(reader, tf_scope):
    """Convert a TF dense layer to PyTorch Linear weights.

    TF: weights [in, out], biases [out]
    PT: weight [out, in], bias [out]
    """
    weights = reader.get_tensor(f'{tf_scope}/weights')
    biases = reader.get_tensor(f'{tf_scope}/biases')
    return {
        'weight': torch.from_numpy(weights.T.copy()).float(),
        'bias': torch.from_numpy(biases.copy()).float(),
    }


def main():
    checkpoint_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'checkpoints')
    checkpoint_path = os.path.join(checkpoint_dir, 'model-17900')
    output_path = os.path.join(checkpoint_dir, 'model.pt')

    print(f'Loading TF checkpoint: {checkpoint_path}')
    reader = tf.train.NewCheckpointReader(checkpoint_path)

    lstm_size = 400
    vocab_size = 73
    num_attn = 10

    # LSTM 1: input = w(73) + stroke(3) = 76
    lstm1_input_size = vocab_size + 3
    lstm1 = convert_lstm_cell(reader,
                              'rnn/LSTMAttentionCell/lstm_cell',
                              lstm1_input_size, lstm_size)

    # Attention linear: input = w(73) + stroke(3) + s1_out(400) = 476
    attn = convert_linear(reader, 'rnn/LSTMAttentionCell/attention')

    # LSTM 2: input = stroke(3) + s1_out(400) + w(73) = 476
    lstm2_input_size = 3 + lstm_size + vocab_size
    lstm2 = convert_lstm_cell(reader,
                              'rnn/LSTMAttentionCell/lstm_cell_1',
                              lstm2_input_size, lstm_size)

    # LSTM 3: input = stroke(3) + s2_out(400) + w(73) = 476
    lstm3_input_size = 3 + lstm_size + vocab_size
    lstm3 = convert_lstm_cell(reader,
                              'rnn/LSTMAttentionCell/lstm_cell_2',
                              lstm3_input_size, lstm_size)

    # GMM output: input = h3(400), output = 121
    gmm = convert_linear(reader, 'rnn/gmm')

    # Build state dict matching model.py structure
    state_dict = {}
    for key, val in lstm1.items():
        state_dict[f'cell.lstm1.{key}'] = val
    for key, val in attn.items():
        state_dict[f'cell.attn_linear.{key}'] = val
    for key, val in lstm2.items():
        state_dict[f'cell.lstm2.{key}'] = val
    for key, val in lstm3.items():
        state_dict[f'cell.lstm3.{key}'] = val
    for key, val in gmm.items():
        state_dict[f'cell.gmm_linear.{key}'] = val

    print(f'\nConverted weights:')
    for name, tensor in sorted(state_dict.items()):
        print(f'  {name}: {list(tensor.shape)}')

    torch.save(state_dict, output_path)
    print(f'\nSaved PyTorch checkpoint: {output_path}')


if __name__ == '__main__':
    main()
