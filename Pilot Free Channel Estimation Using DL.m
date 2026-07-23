clear all; close all; clc;

num_samples = 20000;

x_real = sign(randn(num_samples, 1));
x_imag = sign(randn(num_samples, 1));
x = x_real + 1j * x_imag;

h_real = randn(num_samples, 1);
h_imag = randn(num_samples, 1);
h = (h_real + 1j * h_imag) / sqrt(2);

SNR_dB = 10;
noise_std = 10^(-SNR_dB/20);
noise = (randn(num_samples, 1) + 1j * randn(num_samples, 1)) * noise_std;
y = h .* x + noise;

X_real = real(y);
X_imag = imag(y);
X_train = [X_real, X_imag];

H_real = real(h);
H_imag = imag(h);
Y_train = [H_real, H_imag];

train_idx = 1:15000;
test_idx = 15001:20000;

X_train_data = X_train(train_idx, :);
Y_train_data = Y_train(train_idx, :);
X_test_data = X_train(test_idx, :);
Y_test_data = Y_train(test_idx, :);

layers = [
    featureInputLayer(2)
    fullyConnectedLayer(64)
    reluLayer
    fullyConnectedLayer(64)
    reluLayer
    fullyConnectedLayer(2)
    regressionLayer
    ];

options = trainingOptions('adam', ...
    'MaxEpochs', 20, ...
    'MiniBatchSize', 32, ...
    'Verbose', 1, ...
    'Plots', 'training-progress');

dnn_model = trainNetwork(X_train_data, Y_train_data, layers, options);

h_estimated = predict(dnn_model, X_test_data);
h_estimated_complex = h_estimated(:, 1) + 1j * h_estimated(:, 2);
h_test_complex = Y_test_data(:, 1) + 1j * Y_test_data(:, 2);

mse_loss = mean((h_estimated - Y_test_data).^2, 'all');