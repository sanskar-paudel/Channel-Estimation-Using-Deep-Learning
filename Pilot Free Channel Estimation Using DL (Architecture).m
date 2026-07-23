clear all; close all; clc;
num_samples = 20000;     
SNR_dB = 10;             

x_real_bits = randi([0 1], num_samples, 1);
x_imag_bits = randi([0 1], num_samples, 1);

x_real = 2 * x_real_bits - 1;
x_imag = 2 * x_imag_bits - 1;
x = x_real + 1j * x_imag;

h_real = randn(num_samples, 1);
h_imag = randn(num_samples, 1);
h = (h_real + 1j * h_imag) / sqrt(2);

noise_std = 10^(-SNR_dB/20);
noise = noise_std * (randn(num_samples,1) + 1j*randn(num_samples,1));

y = h .* x + noise;

input_real = real(y);
input_imag = imag(y);
X_data = [input_real input_imag];

channel_real = real(h);
channel_imag = imag(h);
Y_data = [channel_real channel_imag];

train_size = round(0.8 * num_samples);

X_train = X_data(1:train_size, :);
Y_train = Y_data(1:train_size, :);

X_test = X_data(train_size+1:end, :);
Y_test = Y_data(train_size+1:end, :);

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
    'InitialLearnRate', 0.001, ...
    'Verbose', 1, ...
    'Plots', 'training-progress');



dnn_model = trainNetwork(X_train, Y_train, layers, options);

predicted_channel = predict(dnn_model, X_test);

h_estimated = predicted_channel(:,1) + 1j * predicted_channel(:,2);
h_true = Y_test(:,1) + 1j * Y_test(:,2);
mse_value = mean(abs(h_estimated - h_true).^2);

fprintf('\nSimulation Completed Successfully.\n');

figure;
plot(real(h_true(1:100)), 'b');
hold on;
plot(real(h_estimated(1:100)), 'r--');

xlabel('Sample Index');
ylabel('Channel Value');

title('True Channel vs Predicted Channel');
legend('True Channel', 'Predicted Channel');

grid on;
