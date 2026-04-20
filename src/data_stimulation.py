from data_utils.stimulate_data import stimulate_data, plot_samples



length_ts=100
sample_size = 400


# S1: Piecewise Gaussian
# x_train, y_train, tau_train, mu_R_train = stimulate_data(sample_size=700, rho=0, ar_model_name='AR0')

# S1': AR with rho=0.7
# x_train, y_train, tau_train, mu_R_train = stimulate_data(sample_size=700, rho=0.7, ar_model_name='AR1')


# S2: n=100, rho ~ Unif([0, 1]), xi ~ N(0, 2) -> AR
# x_train, y_train, tau_train, mu_R_train = stimulate_data(sample_size=1000, sigma=np.sqrt(2), ar_model_name='ARrho')


# S3: n=100, rho=0, xi ~ Cauchy(0, 0.3)
x_train, y_train, tau_train, mu_R_train = stimulate_data(sample_size=1000, scale=0.3, ar_model_name='ARH')


plot_samples(x_train, y_train, tau_train, num_samples=4)
