clc;
clear;
close all;

%PARAMETERS 
numSubcarriers = 52;
numSymbols = 14;
num_examples = 20000;
SNR_range = 0:5:20;
cp_len = 8;

inputSize = numSubcarriers * numSymbols * 2 + 1; % +1 for SNR
outputSize = numSubcarriers * numSymbols * 2;
fprintf("Generating dataset...\n");

trainData = zeros(num_examples, inputSize);
trainLabels = zeros(num_examples, outputSize);

% UAV PARAMETERS 
K_factor = 4;
velocity = 30;
fc = 2.4e9;
c = 3e8;
doppler_shift = (velocity * fc) / c;

%DATA GENERATION 

for i = 1:num_examples
SNRdB = randi([-5 30]);
    SNR = 10^(SNRdB/10);
    bits = randi([0 1], numSubcarriers, numSymbols, 2);
    x = (2*bits(:,:,1)-1 + 1j*(2*bits(:,:,2)-1))/sqrt(2);

    %OFDM MODULATION 
    tx = ifft(x);
tx_cp = [tx(end-cp_len+1:end,:); tx];

    %UAV RICIAN CHANNEL-NLOS
    los = ones(size(tx_cp));
    nlos = (randn(size(tx_cp)) + 1j*randn(size(tx_cp)))/sqrt(2);

    h = sqrt(K_factor/(K_factor+1))*los +  sqrt(1/(K_factor+1))*nlos;

    % DOPPLER 
    t = 1:numSymbols;
    doppler = exp(1j*2*pi*doppler_shift*t*1e-4);
    h = h .* repmat(doppler, size(h,1), 1);

    %% ----- CHANNEL -----
    rx = tx_cp .* h;
    noise = sqrt(1/(2*SNR)) * ...
        (randn(size(rx)) + 1j*randn(size(rx)));
 y = rx + noise;

    %% ----- RECEIVER -----
    y = y(cp_len+1:end,:);
    y = fft(y);

    input_features = cat(3, real(y), imag(y));
input_vec = reshape(input_features, 1, []);
    snr_feature = SNRdB / 30;

    input_vec = [input_vec snr_feature];

    trainData(i,:) = input_vec;

    h_clean = h(cp_len+1:end,:);

    label = cat(3, real(h_clean), imag(h_clean));

    trainLabels(i,:) = reshape(label,1,[]);

end

fprintf("Dataset ready.\n");

%NORMALIZATION
meanTrain = mean(trainData);
stdTrain = std(trainData) + 1e-8;

trainData = (trainData - meanTrain) ./ stdTrain;

%SPLIT 
idx = randperm(num_examples);
train_ratio = 0.8;
nTrain = floor(train_ratio*num_examples);

XTrain = trainData(idx(1:nTrain),:);
YTrain = trainLabels(idx(1:nTrain),:);
XTest = trainData(idx(nTrain+1:end),:);
YTest = trainLabels(idx(nTrain+1:end),:);

% DNN
layers = [
    featureInputLayer(inputSize,'Normalization','none')

    fullyConnectedLayer(512)
    reluLayer

    fullyConnectedLayer(512)
    reluLayer

    fullyConnectedLayer(256)
    reluLayer

    fullyConnectedLayer(outputSize)

    regressionLayer
];

options = trainingOptions('adam', ...
    'MaxEpochs', 80, ...
    'MiniBatchSize', 32, ...
    'InitialLearnRate', 1e-4, ...
    'Shuffle','every-epoch', ...
    'Plots','training-progress', ...
    'Verbose',1);

fprintf("Training FC-DNN...\n");

net = trainNetwork(XTrain,YTrain,layers,options);

%testing

nmse_dnn = zeros(1,length(SNR_range));
for s = 1:length(SNR_range)
    SNRdB = SNR_range(s);
    SNR = 10^(SNRdB/10);

    nmse_temp = zeros(1,200);

    for k = 1:200

        bits = randi([0 1], numSubcarriers, numSymbols, 2);
        x = (2*bits(:,:,1)-1 + 1j*(2*bits(:,:,2)-1))/sqrt(2);
tx = ifft(x);
        tx_cp = [tx(end-cp_len+1:end,:); tx];

        los = ones(size(tx_cp));
        nlos = (randn(size(tx_cp)) + 1j*randn(size(tx_cp)))/sqrt(2);
 h = sqrt(K_factor/(K_factor+1))*los + sqrt(1/(K_factor+1))*nlos;

        t = 1:numSymbols;
        doppler = exp(1j*2*pi*doppler_shift*t*1e-4);
        h = h .* repmat(doppler,size(h,1),1);

        rx = tx_cp .* h;

        noise = sqrt(1/(2*SNR)) * (randn(size(rx)) + 1j*randn(size(rx)));

        y = rx + noise;

        y = y(cp_len+1:end,:);
        y = fft(y);

        input = cat(3,real(y),imag(y));
        input = reshape(input,1,[]);
        input = [input SNRdB/30];
input = (input - meanTrain) ./ stdTrain;

        h_hat = predict(net,input);
 h_hat = reshape(h_hat,numSubcarriers,numSymbols,2);
        h_hat = h_hat(:,:,1) + 1j*h_hat(:,:,2);
 h_true = h(cp_len+1:end,:);

        nmse_temp(k) = norm(h_hat - h_true,'fro')^2 / norm(h_true,'fro')^2;
 end

    nmse_dnn(s) = mean(nmse_temp);
 fprintf("SNR=%d dB -> NMSE=%.6f\n",SNRdB,nmse_dnn(s));

end

%plot
figure;
semilogy(SNR_range,nmse_dnn,'-o','LineWidth',2);
grid on;

xlabel('SNR (dB)');
ylabel('NMSE');
title('SNR-Aware Pilot-Free UAV-CP-OFDM FC-DNN');