import numpy as np
from scipy.signal import detrend
from scipy.optimize import curve_fit


def rae(x_data, x_time):
    "x_data - An array containing the expression data for one gene at n sampling times"
    "x_time - An array containing the n sampling times "
    # Remove linear trend from time series, mean to 0, slope to 0
    x_detrend = detrend(x_data.values, type='linear')
    # Calculate FFT of the detrended series
    D = np.fft.fftshift(np.fft.fft(x_detrend))
    if (x_detrend.shape[0] % 2) == 0:
        mid = int((x_detrend.shape[0] / 2) + 1)
    else:
        mid = int(np.floor(x_detrend.shape[0] / 2) + 1)

    D = D[mid:mid + 10]  # assumes that dominant frequency is less than or equal to 10
    ind = np.argmax(abs(D))
    # Initialise NLLS fitting using parameter estimates from FFT frequencies
    freq = ind - 1
    phase = np.angle(D[ind])
    amp = np.mean(abs(x_detrend))

    def evaluateModel(model, x_time):
        # Cosine fitting model
        freq = model[0]
        phase = model[1]
        amp = model[2]
        f = amp * np.cos(freq * 2 * np.pi / x_time.shape[0] * x_time + phase)
        return f

    def fit_func(x, freq, phase, amp):
        # Cosine fitting model
        return amp * np.cos(freq * 2 * np.pi / x.shape[0] * x + phase)

    results = []
    for i in range(0, 5):
        # Fit cosine curve to data using different initial parameters
        params, cov_matrix = curve_fit(fit_func, xdata=x_time,
                                       p0=np.array([freq, phase, amp]),
                                       ydata=x_detrend)
        preds = evaluateModel(params, x_time)
        results.append(sum((preds - x_detrend) ** 2))
        freq += 1

    # Refit model using fit with lowest error
    freq = np.argmin(results)
    params, cov_matrix = curve_fit(fit_func, xdata=x_time,
                                   p0=np.array([freq, phase, amp]),
                                   ydata=x_detrend)
    preds = evaluateModel(params, x_time)
    # print("Error", sum((preds - x_detrend) ** 2))
    # print("Amplitude", params[2])
    # Generate confidence interval
    sigma_ab = np.sqrt(np.diagonal(cov_matrix))
    lower = params[2] - 0.975 * sigma_ab[2]
    upper = params[2] + 0.975 * sigma_ab[2]
    # Calculate RAE
    rae_score = np.abs(2 * (upper - lower) / params[2])
    if rae_score > 1:
        rae_score = 1
    # print("RAE", rae_score)
    return rae_score
