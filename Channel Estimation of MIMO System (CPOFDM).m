clc; clear; close all;             
rng(0,'twister');                  

%% ------------------ System Parameters (CP-OFDM & MIMO) ------------------
NT = 64;                            % number of transmit antennas
NR = 2;                             % number of receive antennas
NT_H = 8; NT_V = 8;                 % 8x8 uniform planar transmit array
rho = 0.9;                          % spatial correlation coefficient between antennas
r = 0.875;                          % fraction of antennas used as pilots
Npilot = round(r*NT);               % total number of pilot antennas
Nsc = 128;                          % number of OFDM subcarriers
N_L = round(sqrt(NT));              % number of pilot repetitions for noise averaging
EbN0_dBs = 0:5:20;                  % Eb/N0 range in dB
mod_order = 4;                      % QPSK modulation
bits_per_symbol = log2(mod_order);  % number of bits per QPSK symbol
EsN0_dBs = EbN0_dBs + 10*log10(bits_per_symbol); % convert Eb/N0 to Es/N0

%% ------------------ Correlation Matrix ------------------
R_h = toeplitz(rho.^(0:NT_H-1));     % horizontal spatial correlation matrix
R_v = toeplitz(rho.^(0:NT_V-1));     % vertical spatial correlation matrix
R_t = kron(R_h, R_v);               % full transmit correlation matrix (Kronecker model)
R_t = (R_t + R_t')/2;               % ensure matrix is Hermitian/symmetric
[U,D] = eig(R_t);                   % eigenvalue decomposition
D = max(real(diag(D)),0);           % force eigenvalues to be non-negative
R_t_sqrt = U*diag(sqrt(D))*U';      % compute square-root of correlation matrix

%% ------------------ Pilot Selection ------------------
P_idx = sort(randperm(NT, Npilot)); % randomly choose pilot antenna indices
N_idx = setdiff(1:NT, P_idx);       % remaining antennas are treated as unknown
input_dim = 2*NR*Npilot;            % DNN input size (real + imag parts)
output_dim = 2*NR*(NT-Npilot);      % DNN output size (real + imag parts)

%% ------------------ Channel PDP ------------------
p_db = [0 -1 -9 -10 -15 -20];       % power delay profile (Pedestrian-A)
p_lin = 10.^(p_db/10);              % convert from dB to linear
p_lin = p_lin/sum(p_lin);           % normalize total power
Ntaps = length(p_lin);              % number of channel taps

%% ------------------ Train the DNN ------------------
disp('1/2: Training DNN...');
N_train = 4000;                     % total training samples
X_train = zeros(N_train, input_dim);% input training matrix
Y_train = zeros(N_train, output_dim);% output training matrix

for n = 1:N_train
    EsN0_train = 20 + 10*log10(bits_per_symbol);
    sigma2_train = 1/(10^(EsN0_train/10));
    
   % rand_snr = randi([round(min(EsN0_dBs)), round(max(EsN0_dBs))]);
   % sigma2_train = 1/(10^(rand_snr/10));   % randomly vary SNR during training
    
    % -------- Generate correlated frequency-selective channel --------
    h_time = zeros(NR, NT, Ntaps);
    for l = 1:Ntaps
        Hw = (randn(NR,NT)+1i*randn(NR,NT))/sqrt(2); % i.i.d Rayleigh fading
        h_time(:,:,l) = sqrt(p_lin(l)) * Hw * R_t_sqrt; % apply tap power + spatial correlation
    end
    H_freq = fft(h_time, Nsc, 3);   % convert to frequency domain (OFDM)
    
    k = randi(Nsc);                 % randomly pick one subcarrier
    H_k = H_freq(:,:,k);            % channel at that subcarrier
    
    % -------- Full LMMSE estimation (training uses full pilots) --------
    H_hat_full = zeros(NR, NT);
    for rcv = 1:NR
        y_all = zeros(1, NT);
        for l_p = 1:N_L
            noise = sqrt(sigma2_train/2)*(randn(1,NT)+1i*randn(1,NT));
            y_all = y_all + (H_k(rcv,:) + noise); % received signal
        end
        y_all = y_all/N_L;          % average pilots
        H_hat_full(rcv,:) = (R_t*((R_t + sigma2_train/N_L*eye(NT)) \ y_all.')).';
        % full LMMSE estimation
    end
    
    pilot_part = H_hat_full(:,P_idx);   % estimated pilot antennas
    null_part  = H_hat_full(:,N_idx);   % estimated non-pilot antennas
    
    X_train(n,:) = [real(pilot_part(:)); imag(pilot_part(:))].'; % DNN input
    Y_train(n,:) = [real(null_part(:));  imag(null_part(:))].';  % DNN target
end

% -------- Normalize training data --------
muX = mean(X_train); 
sX = std(X_train)+1e-12; 
muY = mean(Y_train); 
sY = std(Y_train)+1e-12; 

X_norm = (X_train-muX)./sX;
Y_norm = (Y_train-muY)./sY;

% -------- Define DNN architecture --------
layers = [
    featureInputLayer(input_dim,'Normalization','none')
    fullyConnectedLayer(1024)
    reluLayer
    fullyConnectedLayer(1024)
    reluLayer
    fullyConnectedLayer(1024)
    reluLayer
    fullyConnectedLayer(output_dim)
    regressionLayer];

options = trainingOptions('adam', ...
    'InitialLearnRate',1e-4, ...
    'MaxEpochs',40, ...
    'MiniBatchSize',128, ...
    'Verbose',false);

net = trainNetwork(single(X_norm), single(Y_norm), layers, options);
% train neural network

%% ------------------ NMSE Testing ------------------
disp('2/2: NMSE Monte Carlo Testing...');
numSNR = length(EbN0_dBs);
NMSE_conv = zeros(numSNR,1); 
NMSE_dnn  = zeros(numSNR,1);

trials = 500;    % Monte Carlo trials per SNR

for si = 1:numSNR
    
    EsN0_lin = 10^(EsN0_dBs(si)/10);
    sigma2 = 1/EsN0_lin;   % noise variance
    
    nmse_c = 0; 
    nmse_d = 0;
    
    for t = 1:trials
        
        % -------- Generate new channel --------
        h_time = zeros(NR, NT, Ntaps);
        for l = 1:Ntaps
            Hw = (randn(NR,NT)+1i*randn(NR,NT))/sqrt(2);
            h_time(:,:,l) = sqrt(p_lin(l)) * Hw * R_t_sqrt;
        end
        H_freq = fft(h_time, Nsc, 3);
        
        for k = 1:Nsc
            
            H_true = H_freq(:,:,k);   % true channel
            
            %% -------- Conventional Full LMMSE --------
            H_conv = zeros(NR, NT);
            for rcv = 1:NR
                y_all = zeros(1, NT);
                for l_p = 1:N_L
                    noise = sqrt(sigma2/2)*(randn(1,NT)+1i*randn(1,NT));
                    y_all = y_all + (H_true(rcv,:) + noise);
                end
                y_all = y_all/N_L;
                H_conv(rcv,:) = (R_t*((R_t + sigma2/N_L*eye(NT)) \ y_all.')).';
            end
            
            %% -------- Proposed Partial + DNN --------
            H_dnn = zeros(NR, NT);
            H_pilot_est = zeros(NR, Npilot);
            
            for rcv = 1:NR
                y_p = zeros(1, Npilot);
                for l_p = 1:N_L
                    noise_p = sqrt(sigma2/2)*(randn(1,Npilot)+1i*randn(1,Npilot));
                    y_p = y_p + (H_true(rcv,P_idx) + noise_p);
                end
                y_p = y_p/N_L;
                
                Rpp = R_t(P_idx,P_idx);
                H_pilot_est(rcv,:) = (Rpp*((Rpp + sigma2/N_L*eye(Npilot)) \ y_p.')).';
            end
            
            H_dnn(:,P_idx) = H_pilot_est;   % fill pilot estimates
            
            x_test = [real(H_pilot_est(:)); imag(H_pilot_est(:))].';
            x_test = (x_test - muX) ./ sX;  % normalize
            
            y_pred = predict(net, single(x_test)); % DNN prediction
            y_pred = (y_pred .* sY) + muY;         % denormalize
            
            half = output_dim/2;
            H_dnn(:,N_idx) = reshape( ...
                y_pred(1:half)+1i*y_pred(half+1:end), NR, []);
            
            %% -------- NMSE Calculation --------
            nmse_c = nmse_c + norm(H_true - H_conv,'fro')^2 / norm(H_true,'fro')^2;
            nmse_d = nmse_d + norm(H_true - H_dnn,'fro')^2 / norm(H_true,'fro')^2;
        end
    end
    
    NMSE_conv(si) = nmse_c / (trials*Nsc);
    NMSE_dnn(si)  = nmse_d / (trials*Nsc);
end

%% ------------------ Plot ------------------
figure;
semilogy(EbN0_dBs, NMSE_conv, 'b-o', 'LineWidth', 1.5); hold on;
semilogy(EbN0_dBs, NMSE_dnn, 'r-*', 'LineWidth', 1.5);
grid on;
xlabel('E_b/N_0 (dB)');
ylabel('NMSE');
title('NMSE vs SNR (CP-OFDM)');
legend('Conventional (Full LMMSE)', 'Proposed (Partial + DNN)');