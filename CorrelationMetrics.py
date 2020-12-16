import numpy as np
import tqdm

def autocorr(x):
    n = x.size
    norm = (x - np.mean(x))
    result = np.correlate(norm, norm, mode='same')
    acorr = result[n//2 + 1:] / (x.var() * np.arange(n-1, n//2, -1))
    lag = np.abs(acorr).argmax() + 1
    r = acorr[lag-1]
    return r, lag

def get_autocorrelated_genes(J, X_ID):
    clock_genes = []
    indices = []
    scores = []
    for i in tqdm.tqdm(range(J.shape[1])):
        fft = np.fft.rfft(J.iloc[:, i], norm="ortho")
        def abs2(x):
            return x.real**2 + x.imag**2

        r, lag = autocorr(J.iloc[:, i])
        clock_genes.append(X_ID.iloc[i])
        scores.append(r)
        indices.append(i)
    return indices, clock_genes, scores


def cross_corr(y1, y2, X_ID):
  """Calculates the cross correlation and lags without normalization.

  The definition of the discrete cross correlation is in:
  https://www.mathworks.com/help/matlab/ref/xcorr.html

  Args:
    y1, y2: Should have the same length.

  Returns:
    max_corr: Maximum correlation without normalization.
    lag: The lag in terms of the index.
  """

  clock_genes = []
  indices = []
  cross_corrs = []

  for i in tqdm.tqdm(range(y1.shape[1])):
      y1_auto_corr = np.dot(y1.iloc[:, i], y1.iloc[:, i]) / len(y1.iloc[:, i])
      y2_auto_corr = np.dot(y2, y2) / len(y1.iloc[:, i])
      corr = np.correlate(y1.iloc[:, i], y2, mode='same')

      # The unbiased sample size is N - lag.
      unbiased_sample_size = np.correlate(
          np.ones(len(y1.iloc[:, i])), np.ones(len(y1.iloc[:, i])), mode='same')
      corr = corr / unbiased_sample_size / np.sqrt(y1_auto_corr * y2_auto_corr)
      shift = len(y1.iloc[:, i]) // 2

      max_corr = np.max(corr)
      argmax_corr = np.argmax(corr)
      clock_genes.append(X_ID.iloc[i])
      indices.append(i)
      cross_corrs.append(max_corr)
  return indices, clock_genes, cross_corrs


