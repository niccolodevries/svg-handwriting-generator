"""PyTorch implementation of the Graves handwriting synthesis RNN.

Reproduces the TensorFlow model architecture:
- 3-layer LSTM stack with Gaussian mixture attention
- Output: mixture of bivariate Gaussians + Bernoulli end-of-stroke
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class LSTMAttentionCell(nn.Module):
    """Single-step RNN cell with 3 stacked LSTMs and soft attention."""

    def __init__(self, lstm_size=400, num_attn_mixture_components=10,
                 num_output_mixture_components=20, vocab_size=73):
        super().__init__()
        self.lstm_size = lstm_size
        self.num_attn = num_attn_mixture_components
        self.num_output = num_output_mixture_components
        self.vocab_size = vocab_size
        self.output_units = 6 * num_output_mixture_components + 1

        # LSTM 1: input = [w(vocab_size) + stroke(3)]
        self.lstm1 = nn.LSTMCell(vocab_size + 3, lstm_size)
        # Attention: input = [w(vocab_size) + stroke(3) + s1_out(lstm_size)]
        self.attn_linear = nn.Linear(vocab_size + 3 + lstm_size,
                                     3 * num_attn_mixture_components)
        # LSTM 2: input = [stroke(3) + s1_out(lstm_size) + w(vocab_size)]
        self.lstm2 = nn.LSTMCell(3 + lstm_size + vocab_size, lstm_size)
        # LSTM 3: input = [stroke(3) + s2_out(lstm_size) + w(vocab_size)]
        self.lstm3 = nn.LSTMCell(3 + lstm_size + vocab_size, lstm_size)
        # GMM output from h3
        self.gmm_linear = nn.Linear(lstm_size, self.output_units)

    def zero_state(self, batch_size, device='cpu'):
        z = lambda *s: torch.zeros(*s, device=device)
        return {
            'h1': z(batch_size, self.lstm_size),
            'c1': z(batch_size, self.lstm_size),
            'h2': z(batch_size, self.lstm_size),
            'c2': z(batch_size, self.lstm_size),
            'h3': z(batch_size, self.lstm_size),
            'c3': z(batch_size, self.lstm_size),
            'alpha': z(batch_size, self.num_attn),
            'beta': z(batch_size, self.num_attn),
            'kappa': z(batch_size, self.num_attn),
            'w': z(batch_size, self.vocab_size),
        }

    def forward(self, inputs, state, attention_values, attention_lengths, bias=None):
        """One step of the cell.

        Args:
            inputs: [batch, 3] stroke input (dx, dy, eos)
            state: dict of state tensors
            attention_values: [batch, max_chars, vocab_size] one-hot chars
            attention_lengths: [batch] character sequence lengths
            bias: [batch] bias values for sampling neatness

        Returns:
            output: [batch, lstm_size]
            new_state: dict of updated state tensors
        """
        batch_size = inputs.shape[0]
        device = inputs.device
        char_len = attention_values.shape[1]

        # LSTM 1
        s1_in = torch.cat([state['w'], inputs], dim=1)
        h1, c1 = self.lstm1(s1_in, (state['h1'], state['c1']))

        # Attention
        attn_in = torch.cat([state['w'], inputs, h1], dim=1)
        attn_params = F.softplus(self.attn_linear(attn_in))
        alpha, beta, kappa = torch.chunk(attn_params, 3, dim=1)
        kappa = state['kappa'] + kappa / 25.0
        beta = torch.clamp(beta, min=0.01)

        # Compute phi: [batch, num_attn, char_len]
        u = torch.arange(char_len, device=device, dtype=torch.float32)
        u = u.view(1, 1, -1)  # [1, 1, char_len]
        kappa_exp = kappa.unsqueeze(2)  # [batch, num_attn, 1]
        alpha_exp = alpha.unsqueeze(2)
        beta_exp = beta.unsqueeze(2)
        phi = torch.sum(alpha_exp * torch.exp(-torch.square(kappa_exp - u) / beta_exp), dim=1)
        # phi: [batch, char_len]

        # Masked attention window
        seq_mask = torch.arange(char_len, device=device).unsqueeze(0) < attention_lengths.unsqueeze(1)
        seq_mask = seq_mask.float().unsqueeze(2)  # [batch, char_len, 1]
        w = torch.sum(phi.unsqueeze(2) * attention_values * seq_mask, dim=1)  # [batch, vocab_size]

        # LSTM 2
        s2_in = torch.cat([inputs, h1, w], dim=1)
        h2, c2 = self.lstm2(s2_in, (state['h2'], state['c2']))

        # LSTM 3
        s3_in = torch.cat([inputs, h2, w], dim=1)
        h3, c3 = self.lstm3(s3_in, (state['h3'], state['c3']))

        new_state = {
            'h1': h1, 'c1': c1,
            'h2': h2, 'c2': c2,
            'h3': h3, 'c3': c3,
            'alpha': alpha, 'beta': beta, 'kappa': kappa,
            'w': w,
            'phi': phi,
        }
        return h3, new_state

    def output_function(self, state, bias=None):
        """Sample from GMM output distribution.

        Returns: [batch, 3] — (dx, dy, eos)
        """
        params = self.gmm_linear(state['h3'])
        pis, mus, sigmas, rhos, es = self._parse_parameters(params, bias)

        batch_size = pis.shape[0]

        # Sample component index
        idx = torch.multinomial(pis, 1).squeeze(1)  # [batch]

        # Get selected mu and sigma
        mu1, mu2 = mus[:, :self.num_output], mus[:, self.num_output:]
        sigma1, sigma2 = sigmas[:, :self.num_output], sigmas[:, self.num_output:]

        batch_idx = torch.arange(batch_size, device=pis.device)
        sel_mu1 = mu1[batch_idx, idx]
        sel_mu2 = mu2[batch_idx, idx]
        sel_sigma1 = sigma1[batch_idx, idx]
        sel_sigma2 = sigma2[batch_idx, idx]
        sel_rho = rhos[batch_idx, idx]

        # Sample from bivariate Gaussian
        z1 = torch.randn_like(sel_mu1)
        z2 = torch.randn_like(sel_mu2)
        x = sel_mu1 + sel_sigma1 * z1
        y = sel_mu2 + sel_sigma2 * (sel_rho * z1 + torch.sqrt(1 - sel_rho ** 2) * z2)

        # Sample end-of-stroke
        e = torch.bernoulli(es.squeeze(1))

        return torch.stack([x, y, e], dim=1)

    def termination_condition(self, state, attention_lengths, bias=None):
        """Check if generation should stop for each sample in batch."""
        phi = state['phi']  # [batch, char_len]
        char_idx = torch.argmax(phi, dim=1).int()  # [batch]
        final_char = char_idx >= attention_lengths - 1
        past_final_char = char_idx >= attention_lengths
        output = self.output_function(state, bias)
        es = output[:, 2].int()
        is_eos = es == 1
        return (final_char & is_eos) | past_final_char

    def _parse_parameters(self, gmm_params, bias=None, eps=1e-8, sigma_eps=1e-4):
        """Parse GMM parameters from network output."""
        K = self.num_output
        pis = gmm_params[:, :K]
        sigmas = gmm_params[:, K:3*K]
        rhos = gmm_params[:, 3*K:4*K]
        mus = gmm_params[:, 4*K:6*K]
        es = gmm_params[:, 6*K:]

        if bias is not None:
            pis = pis * (1 + bias.unsqueeze(1))
            sigmas = sigmas - bias.unsqueeze(1)

        pis = F.softmax(pis, dim=-1)
        pis = torch.where(pis < 0.01, torch.zeros_like(pis), pis)
        # Renormalize after zeroing small components
        pis = pis / (pis.sum(dim=-1, keepdim=True) + eps)

        sigmas = torch.clamp(torch.exp(sigmas), min=sigma_eps)
        rhos = torch.clamp(torch.tanh(rhos), min=eps - 1.0, max=1.0 - eps)
        es = torch.clamp(torch.sigmoid(es), min=eps, max=1.0 - eps)
        es = torch.where(es < 0.01, torch.zeros_like(es), es)

        return pis, mus, sigmas, rhos, es


class HandwritingSynthesisModel(nn.Module):
    """Full handwriting synthesis model with sampling."""

    def __init__(self, lstm_size=400, num_attn_mixture_components=10,
                 num_output_mixture_components=20, vocab_size=73):
        super().__init__()
        self.cell = LSTMAttentionCell(
            lstm_size=lstm_size,
            num_attn_mixture_components=num_attn_mixture_components,
            num_output_mixture_components=num_output_mixture_components,
            vocab_size=vocab_size,
        )

    @torch.no_grad()
    def sample(self, chars, chars_len, bias, max_steps,
               prime=False, x_prime=None, x_prime_len=None):
        """Generate handwriting strokes for given text.

        Args:
            chars: [batch, max_chars] int tensor of character indices
            chars_len: [batch] int tensor of character lengths
            bias: [batch] float tensor of bias values
            max_steps: int, maximum generation steps
            prime: bool, whether to prime with style strokes
            x_prime: [batch, max_prime_len, 3] style priming strokes
            x_prime_len: [batch] int, lengths of priming strokes

        Returns:
            list of [N, 3] numpy arrays, one per sample
        """
        batch_size = chars.shape[0]
        device = chars.device

        attention_values = F.one_hot(chars.long(), self.cell.vocab_size).float()
        state = self.cell.zero_state(batch_size, device)

        # Prime with style strokes if requested
        if prime and x_prime is not None:
            max_prime_steps = int(x_prime_len.max().item())
            for t in range(max_prime_steps):
                inp = x_prime[:, t, :]  # [batch, 3]
                _, state = self.cell(inp, state, attention_values, chars_len, bias)

        # Free-run sampling
        inp = torch.cat([torch.zeros(batch_size, 2, device=device),
                         torch.ones(batch_size, 1, device=device)], dim=1)

        outputs = []
        finished = torch.zeros(batch_size, dtype=torch.bool, device=device)
        # Track when attention first reaches the final character per sample
        # so we can trim trailing artifacts after that point
        attn_done_step = torch.full((batch_size,), max_steps, dtype=torch.long,
                                    device=device)

        for t in range(max_steps):
            # Save previous state so we can freeze finished samples
            prev_state = {k: v.clone() for k, v in state.items()}

            _, state = self.cell(inp, state, attention_values, chars_len, bias)

            # Track when attention peaks at the final character
            phi = state['phi']  # [batch, char_len]
            attn_pos = torch.argmax(phi, dim=1)  # [batch]
            just_reached_end = (attn_pos >= chars_len - 1) & (attn_done_step == max_steps)
            attn_done_step = torch.where(just_reached_end,
                                         torch.tensor(t, device=device),
                                         attn_done_step)

            # Check termination
            term = self.cell.termination_condition(state, chars_len, bias)
            finished = finished | term

            # Freeze state for finished samples (restore previous state)
            mask = finished.unsqueeze(1)  # [batch, 1]
            for k in state:
                state[k] = torch.where(mask, prev_state[k], state[k])

            if finished.all():
                break

            # Sample next input, zero out finished samples
            inp = self.cell.output_function(state, bias)
            inp = torch.where(mask, torch.zeros_like(inp), inp)
            outputs.append(inp.cpu().numpy())

        if not outputs:
            return [np.zeros((0, 3)) for _ in range(batch_size)]

        # Stack and split by sample
        all_outputs = np.stack(outputs, axis=1)  # [batch, time, 3]
        attn_done_np = attn_done_step.cpu().numpy()

        samples = []
        for i in range(batch_size):
            sample = all_outputs[i]

            # Trim trailing strokes: keep a small grace window after
            # attention reached the last character, then cut
            grace = 15  # allow a few strokes to finish the last letter
            if attn_done_np[i] < max_steps:
                cut_at = attn_done_np[i] + grace
                sample = sample[:cut_at]

            # Remove trailing zeros
            nonzero = ~np.all(sample == 0.0, axis=1)
            if nonzero.any():
                last = np.where(nonzero)[0][-1]
                samples.append(sample[:last + 1])
            else:
                samples.append(sample)
        return samples
