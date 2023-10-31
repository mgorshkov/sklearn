from tabulate import tabulate
import numpy as np

def generate_data(rank, num_points, noise_level):
    np.random.seed(42)
    x = np.linspace(-10, 10, num_points)
    y = np.linspace(-10, 10, num_points)
    if rank == 1:
        z = 3 * x + 5 + noise_level * np.random.randn(num_points)
        data = np.column_stack((x, y, z))
    elif rank == 2:
        z = 2 * x + 3 * y + 5 + noise_level * np.random.randn(num_points)
        data = np.column_stack((x, y, z))
    elif rank == 3:
        z = 2 * x**2 + 3 * y**2 + 5 + noise_level * np.random.randn(num_points)
        data = np.column_stack((x, y, z))
    return data

def GMT_trend2d(data, rank):
    import numpy as np
    from sklearn.linear_model import LinearRegression
    # scale factor for normally distributed data is 1.4826
    # https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.median_abs_deviation.html
    MAD_NORMALIZE = 1.4826
    # significance value
    sig_threshold = 0.51

    if rank not in [1,2,3]:
        raise Exception('Number of model parameters "rank" should be 1, 2, or 3')

    #see gmt_stat.c
    def gmtstat_f_q (chisq1, nu1, chisq2, nu2):
        import scipy.special as sc

        if chisq1 == 0.0:
            return 1
        if chisq2 == 0.0:
            return 0
        return sc.betainc(0.5*nu2, 0.5*nu1, chisq2/(chisq2+chisq1))

    if rank in [2,3]:
        x = data[:,0]
        x = np.interp(x, (x.min(), x.max()), (-1, +1))
    if rank == 3:
        y = data[:,1]
        y = np.interp(y, (y.min(), y.max()), (-1, +1))
    z = data[:,2]
    w = np.ones(z.shape)

    if rank == 1:
        xy = np.expand_dims(np.zeros(z.shape),1)
    elif rank == 2:
        xy = np.expand_dims(x,1)
    elif rank == 3:
        xy = np.stack([x,y]).transpose()

    # create linear regression object
    mlr = LinearRegression()

    chisqs = []
    coeffs = []
    while True:
        # fit linear regression
        mlr.fit(xy, z, sample_weight=w)

        r = np.abs(z - mlr.predict(xy))
        chisq = np.sum((r**2*w))/(z.size-3)
        chisqs.append(chisq)
        k = 1.5 * MAD_NORMALIZE * np.median(r)
        w = np.where(r <= k, 1, (2*k/r) - (k * k/(r**2)))
        sig = 1 if len(chisqs)==1 else gmtstat_f_q(chisqs[-1], z.size-3, chisqs[-2], z.size-3)
        # Go back to previous model only if previous chisq < current chisq
        if len(chisqs)==1 or chisqs[-2] > chisqs[-1]:
            coeffs = [mlr.intercept_, *mlr.coef_]

        #print ('chisq', chisq, 'significant', sig)
        if sig < sig_threshold:
            break

    # get the slope and intercept of the line best fit
    return (coeffs[:rank])

def calculate_mse(data, coeffs, rank):
    z_actual = data[:, 2]

    if rank == 1:
        z_predicted = coeffs[0]
    elif rank == 2:
        # Interpolate x the same way as in GMT_trend2d
        x = data[:, 0]
        x_interp = np.interp(x, (x.min(), x.max()), (-1, +1))
        z_predicted = coeffs[0] + coeffs[1] * x_interp
    elif rank == 3:
        # Interpolate x and y the same way as in GMT_trend2d
        x = data[:, 0]
        x_interp = np.interp(x, (x.min(), x.max()), (-1, +1))
        y = data[:, 1]
        y_interp = np.interp(y, (y.min(), y.max()), (-1, +1))
        z_predicted = coeffs[0] + coeffs[1] * x_interp + coeffs[2] * y_interp

    mse = np.mean((z_actual - z_predicted) ** 2)
    return mse


def test_mse(num_points = 100*1000, ranks = [1, 2, 3], noise_levels = [0, 1, 10, 50]):
    import warnings

    results = []
    # Suppress the specific warning
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        for rank in ranks:
            for noise_level in noise_levels:
                data = generate_data(rank, num_points, noise_level)
                # round the output
                coeffs_gmt = [v.round(8) for v in GMT_trend2d(data, rank)]

                mse_gmt = np.round(calculate_mse(data, coeffs_gmt, rank), 0)

                results.append([rank, noise_level, mse_gmt])

    headers = ["Rank", "Noise Level", "GMT_trend2d, MSE"]
    print(tabulate(results, headers=headers))

test_mse()
