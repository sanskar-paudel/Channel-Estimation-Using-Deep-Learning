clear all; close all; clc;

num_samples = 100000;     
L = 10;
num_blocks = floor(num_samples / L);
SNR_dB = randi([0 20], num_samples, 1);


x_real_bits = randi([0 1], num_samples, 1);
x_imag_bits = randi([0 1], num_samples, 1);
x = (2*x_real_bits - 1) + 1j*(2*x_imag_bits - 1);
x = x / sqrt(2);


h = (randn(num_samples, 1) + 1j*randn(num_samples, 1)) / sqrt(2);
noise_std = 10.^(-SNR_dB/20);
noise = noise_std .* (randn(num_samples, 1) + 1j*randn(num_samples, 1)) / sqrt(2);

y = h .* x + noise;


input_real = real(y);
input_imag = imag(y);

X_blocks_real = reshape(input_real(1:num_blocks*L), L, num_blocks).';
X_blocks_imag = reshape(input_imag(1:num_blocks*L), L, num_blocks).';


X_data = [X_blocks_real, X_blocks_imag]; 


train_size = round(0.8 * num_blocks);

X_train = X_data(1:train_size, :);

Y_train = X_train; 

X_test = X_data(train_size+1:end, :);
Y_test = X_test; 


layers = [
    featureInputLayer(2*L) 
    % ENCODER
    fullyConnectedLayer(128)
    reluLayer
    
    % BOTTLENECK (This layer represents the estimated H)
    fullyConnectedLayer(2, 'Name', 'bottleneck') 
    reluLayer

    % DECODER
    fullyConnectedLayer(128)
    reluLayer
    fullyConnectedLayer(2*L)

    regressionLayer
    ];


options = trainingOptions('adam', ...
    'MaxEpochs', 50, ...
    'MiniBatchSize', 256, ...
    'InitialLearnRate', 0.001, ...
    'Shuffle', 'every-epoch', ...
    'ValidationData', {X_test, Y_test}, ...
    'ValidationFrequency', 300, ...
    'Plots', 'training-progress', ...
    'Verbose', false);


[auto_model, train_info] = trainNetwork(X_train, Y_train, layers, options);

% We predict using the trained model to get the latent channel values
predicted_latent = predict(auto_model, X_test);

h_estimated = predicted_latent(:,1) + 1j * predicted_latent(:,2);

% Get True H for comparison (Mean of H over the block)
h_true_raw = h(train_size*L + 1 : num_blocks*L);
h_true_blocks = mean(reshape(h_true_raw, L, []), 1).';


mse_value = mean(abs(h_estimated - h_true_blocks).^2);
fprintf('Final Pilot-Free MSE = %f\n', mse_value);


figure;
semilogy(train_info.TrainingLoss, 'b', 'LineWidth', 2); hold on;
semilogy(train_info.ValidationLoss, 'r', 'LineWidth', 2);
xlabel('Iterations'); ylabel('Loss (MSE)');
title('Training vs Testing Loss');
legend('Training Loss', 'Testing Loss'); grid on;


figure;
plot(real(h_true_blocks(1:100)), 'b', 'LineWidth', 1.5); hold on;
plot(real(h_estimated(1:100)), 'r--', 'LineWidth', 1.5);
xlabel('Block Index'); ylabel('Channel Value');
title('True Channel vs Autoencoder Estimated Channel');
legend('True Channel', 'Estimated Channel'); grid on;
