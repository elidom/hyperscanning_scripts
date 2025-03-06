clear, clc, close all

% This version of the pipeline calculates the IBS at the level of each
% trial without distinction of speaker-listener directions.

%% Step 1: Load and epoch each dataset
[file_names, file_paths] = uigetfile('ready_data_v0\\*.set',  'set Files (*.set*)','MultiSelect','on');

sub_A = epoch_hyperscanning_data_full_trial(file_names{1}, 'Onset', 'End');
sub_B = epoch_hyperscanning_data_full_trial(file_names{2}, 'Onset', 'End');

if size(sub_A, 1) == size(sub_B, 1)
    if sum(string(sub_A(:,2)) == string(sub_B(:,2))) == size(sub_A, 1)
        disp('# ----------------- Two datasets epoched successfully ----------------- #')
    else
        disp('!!! Error: One or more mismatches found between datasets.')
    end
else
    disp('!!! Error: Datasets are of different size.')
end

%% Step 2: Prepare EEG-only Cell Arrays and Confirm that Trial Length Matches

% Misc Parameters
Nblock =   size(sub_A, 1);
Nch    = size(sub_B{1,1}, 1); 
eeg_fs = 500;

% Initialize arrays
info_A = cell(Nblock, 1);
eeg_A = cell(Nblock, 1);

info_B = cell(Nblock, 1);
eeg_B = cell(Nblock, 1);

% Extract EEG data from trials (SUB A)
for i = 1:Nblock
    % Subject A
    curr_data_temp_A = sub_A{i, 1};                 % Current trial EEG data
    curr_data_A = curr_data_temp_A([1:32 34], :);   % Stim track (channel 65)
    info_A{i} = sub_A{i, 2};                        % Store trial information
    eeg_A{i} = curr_data_A;                        % Store EEG data
    
    % Subject B
    curr_data_temp_B = sub_B{i, 1};                 % Current trial EEG data
    curr_data_B = curr_data_temp_B(1:33, :);   % Stim track (channel 65)
    info_B{i} = sub_B{i, 2};                        % Store trial information
    eeg_B{i} = curr_data_B;                        % Store EEG data
end

temp_count = 0;
for s = 1:Nblock
    size1 = size(eeg_A{s},2);
    size2 = size(eeg_B{s},2);
    if size1 == size2
        temp_count = temp_count + 1;
    end
end
if temp_count == Nblock
    disp('# - - - - - - - - - Trial length matches across datasets! - - - - - - - - - #')
end


%% Step 3: Apply Filters and Compute Gradients

% % BUTTER
% Compute filter order
% fs = 500;               % Sampling frequency in Hz
% Rp = 3;                 % Passband ripple (dB)
% Rs = 40;                % Stopband attenuation (dB)
% 
% % Alpha FB
% Wp_alpha = [8 12] / (fs/2);   % Passband (normalized)
% Ws_alpha = [6 14] / (fs/2);   % Stopband (normalized)
% 
% [n, Wn] = buttord(Wp_alpha, Ws_alpha, Rp, Rs);
% 
% N = 3;
% [B, A] = butter(N, Wn, 'bandpass'); % Butterworth bandpass filter

% freqz(B,A, [], 500)

% % FIR
fs = 500;         % Sampling frequency in Hz
order = 100;      % Filter order (adjust as needed)

% Calculate normalized cutoff frequencies (between 0 and 1, where 1 corresponds to the Nyquist frequency)
low_cutoff = 8 / (fs/2);    % Lower cutoff (8 Hz)
high_cutoff = 12 / (fs/2);  % Upper cutoff (12 Hz)

% Design the FIR bandpass filter using fir1 (which uses a Hamming window by default)
b = fir1(order, [low_cutoff high_cutoff], 'bandpass');

% diagnose
figure;
freqz(b, 1, 1024, fs);
title('Frequency Response of the Alpha Band FIR Filter');


% Prepare for filtering EEG and audio signals between 2-10 Hz
alpha_A   = cell(Nblock, 1);
d_alpha_A = cell(Nblock, 1);
alpha_B   = cell(Nblock, 1);
d_alpha_B = cell(Nblock, 1);


% Filter the data and compute gradients
for bi = 1:Nblock
    % Filter the sub-A
%     filteredData = filtfilt(B, A, double(eeg_A{bi})');
    filteredData = filtfilt(b, 1, double(eeg_A{bi})');
    alpha_A{bi} = filteredData';
    
    % Filter the sub-B
%     filteredData = filtfilt(B, A, double(eeg_B{bi})');
    filteredData = filtfilt(b, 1, double(eeg_B{bi})');
    alpha_B{bi} = filteredData';
    
    % optional, add gradient to 2d calculation (this could smooth out filter oscilations)
    d_alpha_A{bi} = gradient_dim1(alpha_A{bi});
    d_alpha_B{bi} = gradient_dim1(alpha_B{bi});
end


disp('# -- Alpha done -- #')

%% BUTTER BETA
% Compute filter order
fs = 500;               % Sampling frequency in Hz
Rp = 3;                 % Passband ripple (dB)
Rs = 40;                % Stopband attenuation (dB)
% 

Wp_beta = [13 30] / (fs/2);   % Passband (normalized)
Ws_beta = [11 33] / (fs/2);   % Stopband (normalized)
% 
[n, Wn] = buttord(Wp_beta, Ws_beta, Rp, Rs);
% 
N = 6;
[B, A] = butter(N, Wn, 'bandpass'); 

% diagnose
figure;
freqz(B, A, 1024, fs);
title('Frequency Response of the Beta Band Butter Filter');

% Prepare for filtering EEG and audio signals between 13-30 Hz
beta_A   = cell(Nblock, 1);
d_beta_A = cell(Nblock, 1);
beta_B   = cell(Nblock, 1);
d_beta_B = cell(Nblock, 1);

% Filter the data and compute gradients
for bi = 1:Nblock
    % Filter the sub-A
    filteredData = filtfilt(B, A, double(eeg_A{bi})');
    beta_A{bi} = filteredData';
    
    % Filter the sub-B
    filteredData = filtfilt(B, A, double(eeg_B{bi})');
    beta_B{bi} = filteredData';
    
    % optional, add gradient to 2d calculation (this could smooth out filter oscilations)
    d_beta_A{bi} = gradient_dim1(beta_A{bi});
    d_beta_B{bi} = gradient_dim1(beta_B{bi});
end


disp('# -- Beta done -- #')


%% Step 4: Alpha IBS
tic;
results = nan(Nblock, Nch, Nch);                    % Initialize results matrix

parfor bi = 1:Nblock                                   % Loop for each trial

    brain_A = alpha_A{bi};                          % get trial N from subject A
    brain_B = alpha_B{bi};                          % get trial N from subject B

    dbrain_A = d_alpha_A{bi};
    dbrain_B = d_alpha_B{bi};

    for chi = 1:Nch                                 % Loop for each of subject A's channels

        chan_A  = brain_A(chi,:);                    % fetch info from A's channel N
        dchan_A = dbrain_A(chi,:);

        for cha = 1:Nch                             % Loop for each of subject B's channels

            chan_B = brain_B(cha,:);                % fetch info from B's channel N
            dchan_B = dbrain_B(cha,:);

            sync = gcmi_cc([chan_A, dchan_A], [chan_B, dchan_B]);         % compute gcmi the easy way

            results(bi, chi, cha) = sync;           % Add the MI value to results matrix
        end
    end
end
elapsedTime = toc;
fprintf('Elapsed time: %.4f seconds\n', elapsedTime);
% to do:
% - implement timelags, because why not - could be a good proof of concept.


%% Step 4: Beta IBS
tic;
results_beta = nan(Nblock, Nch, Nch);                    % Initialize results matrix

for bi = 1:Nblock                                   % Loop for each trial

    brain_A = beta_A{bi};                          % get trial N from subject A
    brain_B = beta_B{bi};                          % get trial N from subject B

    for chi = 1:Nch                                 % Loop for each of subject A's channels

        chan_A = brain_A(chi,:);                    % fetch info from A's channel N

        for cha = 1:Nch                             % Loop for each of subject B's channels

            chan_B = brain_B(cha,:);                % fetch info from B's channel N

            sync = gcmi_cc(chan_A, chan_B);         % compute gcmi the easy way

            results_beta(bi, chi, cha) = sync;           % Add the MI value to results matrix
        end
    end
end
elapsedTime = toc;
fprintf('Elapsed time: %.4f seconds\n', elapsedTime);
% to do:
% - implement timelags, because why not - could be a good proof of concept.

%% With time lags

lags = -400:4:400;
Nlags = length(lags);
lagtime = lags ./ eeg_fs;    % lags to time in seconds
L_max = max(abs(lags));

tic;
results = nan(Nblock, Nlags, Nch, Nch);                    % Initialize results matrix

parfor bi = 1:Nblock                                   % Loop for each trial

    brain_A = alpha_A{bi};                          % get trial N from subject A
    brain_B = alpha_B{bi};                          % get trial N from subject B

    for li = 1:Nlags

        l = lags(li);

         % align stim and resp based on lag
        if l == 0

            % Zero lag
            samples2trim = L_max;

            idx_start = ceil(samples2trim/2) + 1;
            idx_end = length(brain_A) - floor(samples2trim/2);

            Alag = brain_A(:,idx_start:idx_end);
            Blag = brain_B(:,idx_start:idx_end);

        elseif l < 0
            % Negative lag: stimulus before response
            lag_abs = abs(l);
            samples2trim = L_max - lag_abs;

            idx_start = ceil(samples2trim/2) + 1;
            idx_end = length(brain_A) - floor(samples2trim/2);

            A_segment = brain_A(:,idx_start:idx_end);
            B_segment = brain_B(:,idx_start:idx_end);

            Alag = A_segment(:, 1:end - lag_abs);
            Blag = B_segment(:, lag_abs + 1:end);

%             size(Alag)

        else % l > 0
            % Positive lag: response before stimulus
            samples2trim = L_max - l;
            
            idx_start = ceil(samples2trim/2) + 1;
            idx_end = length(brain_A) - floor(samples2trim/2);

            A_segment = brain_A(:,idx_start:idx_end);
            B_segment = brain_B(:,idx_start:idx_end);

            Alag = A_segment(:, l + 1:end);
            Blag = B_segment(:, 1:end - l);

        end

        for chi = 1:Nch                                 % Loop for each of subject A's channels
    
            chan_A  = Alag(chi,:);                    % fetch info from A's channel N
    
            for cha = 1:Nch                             % Loop for each of subject B's channels
    
                chan_B = Blag(cha,:);                % fetch info from B's channel N
                 
                sync = gcmi_cc(chan_A, chan_B);         % compute gcmi the easy way
    
                results(bi, li, chi, cha) = sync;           % Add the MI value to results matrix
            end
        end
    end
    disp(['Trial ' num2str(bi), '/' num2str(Nblock) ' done!'])
end
elapsedTime = toc;
fprintf('Elapsed time: %.4f seconds\n', elapsedTime);
% 18 hours with no par

%% Time lagged results

% Flip matrix over the time axis
% results_flipped = fliplr(lagmi3d);

% misc info
channels = sub_B{1,3};
listening_idx    = [1, 5, 9, 15, 19, 23];
low_interest_idx = [2, 6, 10, 13, 16, 20];
high_interest_idx= [3, 7, 11, 12, 17, 21];
silence_idx      = [4, 8, 14, 18, 22];
frontal_idx        = [1,2,3,4,30,31,32];            % Fp1, Fz, F3, F7, F4, F8, Fp2
left_temporal_idx  = [5,9,10];                       % FT9, T7, TP9
central_idx        = [8,6,7,28,29,24,25,33];          % C3, FC5, FC1, FC6, FC2, Cz, C4, FCz
right_temporal_idx = [27,26,21];                     % FT10, T8, TP10
parietal_idx       = [11,12,13,14,15,19,20,22,23];     % CP5, CP1, Pz, P3, P7, P4, P8, CP6, CP2
occipital_idx      = [16,17,18];                     % O1, Oz, O2
new_order = [frontal_idx, left_temporal_idx, central_idx, right_temporal_idx, parietal_idx, occipital_idx];
channels_ordered      = channels(new_order);

% ROI
for c = parietal_idx
    % labels_to_find = c;
    % indices = find(ismember(channels, labels_to_find));
    
    indices = c;
    high_interest_lagged = results(7,:,indices,indices);
    
    % high_interest_chan_lagged = squeeze(mean(mean(high_interest_lagged,3),2)); % left temporal
    
    % Define time axis
    time_lags = -400:4:400;
    
    % Initialize figure
    figure(1), clf
    
    % Plot just the mean
    plot(time_lags, high_interest_lagged, 'k', 'LineWidth', 2)
    
    xlabel('lag (ms)', 'FontSize', 12);
    ylabel('bits', 'FontSize', 12);
    title(channels(c), 'FontSize', 14, 'FontWeight', 'bold');
    pause(2)
end

% to do, look at associations between non-corresponiding electrodes based
% on surrogate results.

%% Explore results of averaged lags

% iff results contains time lags
around_zero = lags(95:107); % -24 to 24 ms
results_around_zero = squeeze(max(results(:,95:107,:,:), [], 2));
% results_around_zero = squeeze(prctile(results(:,95:107,:,:), 95, 2));

avg_mat_listening     = squeeze(mean(results_around_zero(listening_idx,:,:),1))/2;
avg_mat_low_interest  = squeeze(mean(results_around_zero(low_interest_idx,:,:),1));
avg_mat_high_interest = squeeze(mean(results_around_zero(high_interest_idx,:,:),1));
avg_mat_silence       = squeeze(mean(results_around_zero(silence_idx,:,:),1));

% --- Order channels by MI density (highest to lowest) for each condition ---
% For each condition, compute the average MI for each channel (across columns),
% then sort channels in descending order.

% Listening
[~, order_listening] = sort(mean(avg_mat_listening,1), 'descend');
avg_mat_listening = flipud(avg_mat_listening(order_listening, order_listening));
channels_listening = channels(order_listening);

% Low Interest
[~, order_low] = sort(mean(avg_mat_low_interest,1), 'descend');
avg_mat_low_interest = flipud(avg_mat_low_interest(order_low, order_low));
channels_low = channels(order_low);

% High Interest
[~, order_high] = sort(mean(avg_mat_high_interest,1), 'descend');
avg_mat_high_interest = flipud(avg_mat_high_interest(order_high, order_high));
channels_high = channels(order_high);

% Silence
[~, order_silence] = sort(mean(avg_mat_silence,1), 'descend');
avg_mat_silence = flipud(avg_mat_silence(order_silence, order_silence));
channels_silence = channels(order_silence);

% --- Compute a uniform color scale across all conditions ---
all_values = [avg_mat_listening(:); avg_mat_low_interest(:); avg_mat_high_interest(:); avg_mat_silence(:)];
cmin = quantile(all_values, 0.05);
cmax = quantile(all_values, 0.95);

figure(1), clf

% Listening condition
subplot(2,2,1)
imagesc(avg_mat_listening);
caxis([cmin cmax]);       % Uniform color scale
colorbar;
title('Listening');
xlabel('Subject B Channels');
ylabel('Subject A Channels');
set(gca, 'XTick', 1:length(channels_listening), 'XTickLabel', channels_listening, 'XTickLabelRotation', 90);
set(gca, 'YTick', 1:length(channels_listening), 'YTickLabel', flip(channels_listening));

% Silence condition
subplot(2,2,2)
imagesc(avg_mat_silence);
caxis([cmin cmax]);       % Uniform color scale
colorbar;
title('Silence');
xlabel('Subject B Channels');
ylabel('Subject A Channels');
set(gca, 'XTick', 1:length(channels_silence), 'XTickLabel', channels_silence, 'XTickLabelRotation', 90);
set(gca, 'YTick', 1:length(channels_silence), 'YTickLabel', flip(channels_silence));

% Low Interest condition
subplot(2,2,3)
imagesc(avg_mat_low_interest);
caxis([cmin cmax]);       % Uniform color scale
colorbar;
title('Low Interest');
xlabel('Subject B Channels');
ylabel('Subject A Channels');
set(gca, 'XTick', 1:length(channels_low), 'XTickLabel', channels_low, 'XTickLabelRotation', 90);
set(gca, 'YTick', 1:length(channels_low), 'YTickLabel', flip(channels_low));

% High Interest condition
subplot(2,2,4)
imagesc(avg_mat_high_interest);
caxis([cmin cmax]);       % Uniform color scale
colorbar;
title('High Interest');
xlabel('Subject B Channels');
ylabel('Subject A Channels');
set(gca, 'XTick', 1:length(channels_high), 'XTickLabel', channels_high, 'XTickLabelRotation', 90);
set(gca, 'YTick', 1:length(channels_high), 'YTickLabel', flip(channels_high));

% Note how P4 is lit only in both conversation conditions

%% Viz (BETA)
channels = sub_B{1,3};

% Define trial indices for each condition
listening_idx    = [1, 5, 9, 15, 19, 23];
low_interest_idx = [2, 6, 10, 13, 16, 20];
high_interest_idx= [3, 7, 11, 12, 17, 21];
silence_idx      = [4, 8, 14, 18, 22];

% Average matrices for each condition (from your code)
avg_mat_listening     = squeeze(mean(results_beta(listening_idx,:,:),1))/2;
avg_mat_low_interest  = squeeze(mean(results_beta(low_interest_idx,:,:),1));
avg_mat_high_interest = squeeze(mean(results_beta(high_interest_idx,:,:),1));
avg_mat_silence       = squeeze(mean(results_beta(silence_idx,:,:),1));

% --- Order channels by MI density (highest to lowest) for each condition ---
% For each condition, compute the average MI for each channel (across columns),
% then sort channels in descending order.

% Listening
[~, order_listening] = sort(mean(avg_mat_listening,1), 'descend');
avg_mat_listening = flipud(avg_mat_listening(order_listening, order_listening));
channels_listening = channels(order_listening);

% Low Interest
[~, order_low] = sort(mean(avg_mat_low_interest,1), 'descend');
avg_mat_low_interest = flipud(avg_mat_low_interest(order_low, order_low));
channels_low = channels(order_low);

% High Interest
[~, order_high] = sort(mean(avg_mat_high_interest,1), 'descend');
avg_mat_high_interest = flipud(avg_mat_high_interest(order_high, order_high));
channels_high = channels(order_high);

% Silence
[~, order_silence] = sort(mean(avg_mat_silence,1), 'descend');
avg_mat_silence = flipud(avg_mat_silence(order_silence, order_silence));
channels_silence = channels(order_silence);

% --- Compute a uniform color scale across all conditions ---
all_values = [avg_mat_listening(:); avg_mat_low_interest(:); avg_mat_high_interest(:); avg_mat_silence(:)];
cmin = quantile(all_values, 0.05);
cmax = quantile(all_values, 0.95);

figure(1), clf

% Listening condition
subplot(2,2,1)
imagesc(avg_mat_listening);
caxis([cmin cmax]);       % Uniform color scale
colorbar;
title('Listening');
xlabel('Subject B Channels');
ylabel('Subject A Channels');
set(gca, 'XTick', 1:length(channels_listening), 'XTickLabel', channels_listening, 'XTickLabelRotation', 90);
set(gca, 'YTick', 1:length(channels_listening), 'YTickLabel', flip(channels_listening));

% Silence condition
subplot(2,2,2)
imagesc(avg_mat_silence);
caxis([cmin cmax]);       % Uniform color scale
colorbar;
title('Silence');
xlabel('Subject B Channels');
ylabel('Subject A Channels');
set(gca, 'XTick', 1:length(channels_silence), 'XTickLabel', channels_silence, 'XTickLabelRotation', 90);
set(gca, 'YTick', 1:length(channels_silence), 'YTickLabel', flip(channels_silence));

% Low Interest condition
subplot(2,2,3)
imagesc(avg_mat_low_interest);
caxis([cmin cmax]);       % Uniform color scale
colorbar;
title('Low Interest');
xlabel('Subject B Channels');
ylabel('Subject A Channels');
set(gca, 'XTick', 1:length(channels_low), 'XTickLabel', channels_low, 'XTickLabelRotation', 90);
set(gca, 'YTick', 1:length(channels_low), 'YTickLabel', flip(channels_low));

% High Interest condition
subplot(2,2,4)
imagesc(avg_mat_high_interest);
caxis([cmin cmax]);       % Uniform color scale
colorbar;
title('High Interest');
xlabel('Subject B Channels');
ylabel('Subject A Channels');
set(gca, 'XTick', 1:length(channels_high), 'XTickLabel', channels_high, 'XTickLabelRotation', 90);
set(gca, 'YTick', 1:length(channels_high), 'YTickLabel', flip(channels_high));

%% Difference between High and Low Interest conditions
diff_HighMinusSil = avg_mat_high_interest - avg_mat_silence;

cmin = 0;
cmax = quantile(diff_HighMinusSil(:), 0.8);

figure(1);
imagesc(diff_HighMinusSil);
caxis([cmin cmax]);
colorbar;
title('High Interest minus Silence');
xlabel('Subject B Channels');
ylabel('Subject A Channels');
set(gca, 'XTick', 1:length(channels_ordered), 'XTickLabel', channels_ordered, 'XTickLabelRotation', 90);
set(gca, 'YTick', 1:length(channels_ordered), 'YTickLabel', flip(channels_ordered));


diff_LowMinusSil = avg_mat_low_interest - avg_mat_silence;
figure(2);
imagesc(diff_LowMinusSil);
caxis([cmin cmax]);
colorbar;
title('Low Interest minus Silence');
xlabel('Subject B Channels');
ylabel('Subject A Channels');
set(gca, 'XTick', 1:length(channels_ordered), 'XTickLabel', channels_ordered, 'XTickLabelRotation', 90);
set(gca, 'YTick', 1:length(channels_ordered), 'YTickLabel', flip(channels_ordered));

diff_HighMinusLow = avg_mat_high_interest - avg_mat_low_interest;
figure(3);
imagesc(diff_HighMinusLow);
caxis([cmin cmax]);
colorbar;
title('High Interest minus Low Interest');
xlabel('Subject B Channels');
ylabel('Subject A Channels');
set(gca, 'XTick', 1:length(channels_ordered), 'XTickLabel', channels_ordered, 'XTickLabelRotation', 90);
set(gca, 'YTick', 1:length(channels_ordered), 'YTickLabel', flip(channels_ordered));



%% Compute global synchrony for each trial by averaging over channels
% global_sync = squeeze(mean(mean(results, 3), 2));

fz = mean(mean(results_around_zero(:,parietal_idx,parietal_idx),3),2);

% Prepare grouping and data for boxplot
groups = [repmat({'Listening'}, length(listening_idx), 1);
          repmat({'Low Interest'}, length(low_interest_idx), 1);
          repmat({'High Interest'}, length(high_interest_idx), 1);
          repmat({'Silence'}, length(silence_idx), 1)];
data = [fz(listening_idx)'; fz(low_interest_idx)';
        fz(high_interest_idx)'; [fz(silence_idx)' mean(fz(silence_idx))]];

groups = [repmat({'Listening'}, length(listening_idx), 1);
          repmat({'Low Interest'}, length(low_interest_idx), 1);
          repmat({'High Interest'}, length(high_interest_idx), 1);
          repmat({'Silence'}, 6, 1)];

figure;
boxplot(data, groups);
ylabel('Average MI');
title('Interbrain Synchrony by Condition and ROI');


%% Circular Shifting Surrogate Analysis (Alpha)

tic;
Nsurrogate = 200;  % N of iterations
results_sur = nan(Nblock, Nch, Nch, Nsurrogate);  % Initialize surrogate matrix

parfor bi = 1:Nblock
    brain_A = alpha_A{bi};
    dbrain_A = d_alpha_A{bi};

    brain_B = alpha_B{bi};
    dbrain_B = d_alpha_B{bi};

    T = size(brain_B,2);  % N of timepoints in trial

    for surr = 1:Nsurrogate
         % random shift (starting at 20 seconds shift)
         shift_amount = randi([10000, T-10000]);
         
         % apply circular shift to all of B's channels
         brain_B_shift = circshift(brain_B, [0, shift_amount]); % zero up-down, N left-right
         d_brain_B_shift = circshift(dbrain_B, [0, shift_amount]); % zero up-down, N left-right
         
         for chi = 1:Nch
             chan_A = brain_A(chi,:);
             dchan_A = dbrain_A(chi,:);
             
             for cha = 1:Nch
                  chan_B_shift = brain_B_shift(cha,:);
                  dchan_B_shift = d_brain_B_shift(cha,:);
                  sync_sur = gcmi_cc(chan_A, chan_B_shift);
                  results_sur(bi, chi, cha, surr) = sync_sur;
             end
         end
    end
end

elapsedTime = toc;
fprintf('Elapsed time: %.4f seconds\n', elapsedTime);
% to-do: make more efficient by calculating the copula beforehand? Check
% if it can be done before shifting though
% took 18 hours

%% Circular Shifting Surrogate Analysis (Beta)

tic;
Nsurrogate = 200;  % N of iterations
results_sur_beta = nan(Nblock, Nch, Nch, Nsurrogate);  % Initialize surrogate matrix

parfor bi = 1:Nblock
    brain_A = beta_A{bi};
    brain_B = beta_B{bi};
    T = size(brain_B,2);  % N of timepoints in trial

    for surr = 1:Nsurrogate
         % random shift (starting at 20 seconds shift)
         shift_amount = randi([10000, T-10000]);
         
         % apply circular shift to all of B's channels
         brain_B_shift = circshift(brain_B, [0, shift_amount]); % zero up-down, N left-right
         
         for chi = 1:Nch
             chan_A = brain_A(chi,:);
             
             for cha = 1:Nch
                  chan_B_shift = brain_B_shift(cha,:);
                  sync_sur = gcmi_cc(chan_A, chan_B_shift);
                  results_sur_beta(bi, chi, cha, surr) = sync_sur;
             end
         end
    end
end

elapsedTime = toc;
fprintf('Elapsed time: %.4f seconds\n', elapsedTime);
% to-do: make more efficient by calculating the copula beforehand? Check
% if it can be done before shifting though
% took 18 hours

%% Circular Shifting Surrogate Analysis (Alpha, with Time Lags)

tic;
lags = -24:4:24; % if change, change below as well
Nlags = length(lags);
lagtime = lags ./ eeg_fs;    % lags to time in seconds
L_max = max(abs(lags));
Nsurrogate = 100;  % N of iterations
results_sur = nan(Nblock, Nlags, Nch, Nch, Nsurrogate);  % Initialize surrogate matrix

parfor bi = 1:Nblock
    brain_A = alpha_A{bi};
    brain_B = alpha_B{bi};
   
    T = size(brain_B,2);  % N of timepoints in trial

    for surr = 1:Nsurrogate
         % random shift (starting at 20 seconds shift)
         shift_amount = randi([10000, T-10000]);
         
         % apply circular shift to all of B's channels
         brain_B_shift = circshift(brain_B, [0, shift_amount]); % zero up-down, N left-right
             
         for li = 1:Nlags

            lags = -24:4:24;
            l = lags(li);

            if l == 0

                % Zero lag
                samples2trim = L_max;
    
                idx_start = ceil(samples2trim/2) + 1;
                idx_end = length(brain_A) - floor(samples2trim/2);
    
                Alag = brain_A(:,idx_start:idx_end);
                Blag = brain_B_shift(:,idx_start:idx_end);
    
            elseif l < 0
                % Negative lag: stimulus before response
                lag_abs = abs(l);
                samples2trim = L_max - lag_abs;
    
                idx_start = ceil(samples2trim/2) + 1;
                idx_end = length(brain_A) - floor(samples2trim/2);
    
                A_segment = brain_A(:,idx_start:idx_end);
                B_segment = brain_B_shift(:,idx_start:idx_end);
    
                Alag = A_segment(:, 1:end - lag_abs);
                Blag = B_segment(:, lag_abs + 1:end);
    
    %             size(Alag)
    
            else % l > 0
                % Positive lag: response before stimulus
                samples2trim = L_max - l;
                
                idx_start = ceil(samples2trim/2) + 1;
                idx_end = length(brain_A) - floor(samples2trim/2);
    
                A_segment = brain_A(:,idx_start:idx_end);
                B_segment = brain_B_shift(:,idx_start:idx_end);
    
                Alag = A_segment(:, l + 1:end);
                Blag = B_segment(:, 1:end - l);
    
            end

             for chi = 1:Nch
                 chan_A = Alag(chi,:);
                 
                 for cha = 1:Nch
                      chan_B_shift = Blag(cha,:);
                      sync_sur = gcmi_cc(chan_A, chan_B_shift);
                      results_sur(bi, li, chi, cha, surr) = sync_sur;
                 end
             end
         end
    end
    disp(['Trial ' num2str(bi), '/' num2str(Nblock) ' done!'])
end

elapsedTime = toc;
fprintf('Elapsed time: %.4f seconds\n', elapsedTime);
% to-do: make more efficient by calculating the copula beforehand? Check
% if it can be done before shifting though
% took 42 hours in Lab-1
% took ___ hours in Lab-2 with twice as many time lags
%% compute p-values for each channel pair per trial

% lagged_surr_stats = squeeze(max(results_sur, [], 2));
lagged_surr_stats = squeeze(prctile(results_sur, 95, 2));

p_values = nan(Nblock, Nch, Nch);

for bi = 1:Nblock
    for chi = 1:Nch
         for cha = 1:Nch
              observed = results_around_zero(bi,chi,cha);
              surrogate_distribution = squeeze(lagged_surr_stats(bi,chi,cha,:));
              p_values(bi,chi,cha) = sum(surrogate_distribution >= observed) / Nsurrogate;
         end
    end
end

%% Plot p-values

trial_to_plot = 7;
p_trial = squeeze(p_values(trial_to_plot,:,:));  % p-values matrix for trial 1 (Nch x Nch)

% Define significance threshold
sig_threshold = 0.01;

% Create a masked version of the p-values:
% Set non-significant values (p >= threshold) to NaN so they can be made transparent.
p_sig = p_trial;
p_sig(p_trial >= sig_threshold) = 1;

% Plotting the p-values heatmap
figure;
imagesc(p_sig);
colormap('jet');                % Choose a colormap
colorbar;
% Force the color axis to span from 0 to the threshold, so that only significant values are colored.
caxis([0 sig_threshold]);

% Add channel labels on axes (assuming channels_ordered contains your channel names)
set(gca, 'XTick', 1:length(channels_ordered), 'XTickLabel', channels_ordered, ...
         'XTickLabelRotation', 90);
set(gca, 'YTick', 1:length(channels_ordered), 'YTickLabel', channels_ordered);

title(sprintf('Significant p-values (p < %.2f) for trial %d', sig_threshold, trial_to_plot));
xlabel('Subject B Channels');
ylabel('Subject A Channels');


%% Define trial indices for each condition
listening_idx    = [1, 5, 9, 15, 19, 23];
low_interest_idx = [2, 6, 10, 13, 16, 20];
high_interest_idx= [3, 7, 11, 12, 17, 21];
silence_idx      = [4, 8, 14, 18, 22];

% Conditions and corresponding trial indices
conditions = {'Listening', 'Low Interest', 'High Interest', 'Silence'};
cond_indices = {listening_idx, low_interest_idx, high_interest_idx, silence_idx};

%% Combine surrogate distributions across trials for each condition
% Preallocate a cell array to hold p-value matrices (Nch x Nch) for each condition.
p_combined_conditions = cell(1, numel(conditions));

for c = 1:numel(conditions)
    idx = cond_indices{c};       % Trial indices for current condition
    p_combined = nan(Nch, Nch);    % Preallocate for current condition
    
    for i = 1:Nch
        for j = 1:Nch
            % Compute the observed MI for channel pair (i,j) as the average
            MI_obs_mean = mean(results_around_zero(idx, i, j));
            
            % Pool surrogate MI values for the same channel pair across trials
            % results_sur(idx, i, j, :) has dimensions [num_trials x 1 x 1 x Nsurrogate]
%             surrogate_values = reshape(lagged_surr_stats(idx, i, j, :), [], 1);
            surrogate_values = squeeze(mean(lagged_surr_stats(idx, i, j, :), 1));

            
            % Compute the p-value: fraction of surrogate MI values >= observed MI
            p_combined(i, j) = sum(surrogate_values >= MI_obs_mean) / numel(surrogate_values);
        end
    end
    p_combined_conditions{c} = p_combined;
end

% To do: average surrogate instead

%% Plotting the combined p-values as heatmaps
% We'll plot one heatmap per condition. Only channel pairs with p < sig_threshold
% will be colored (others are rendered transparent).

sig_threshold = 0.01;  % Define significance threshold

figure; clf
for c = 1:numel(conditions)
    subplot(2,2,c)
    p_mat = p_combined_conditions{c};
    
    % Create a masked matrix: values not meeting significance are set to NaN.
    p_sig = p_mat;
    p_sig(p_mat >= sig_threshold) = 1;
    
    % Plot the masked matrix using imagesc
    imagesc(p_sig);
    colormap('jet');   % Or any colormap you prefer
    colorbar;
    caxis([0 sig_threshold+.01]);  % Color axis spans only significant p-values
    
    % Use AlphaData to make non-significant entries transparent
%     alphaData = ~isnan(p_sig);
%     set(gca, 'AlphaData', alphaData);
    
    % Set channel labels on the axes.
    % (Assuming channels_ordered is a cell array of channel names that matches Nch.)
    set(gca, 'XTick', 1:length(channels_ordered), 'XTickLabel', channels_ordered, 'XTickLabelRotation', 90);
    set(gca, 'YTick', 1:length(channels_ordered), 'YTickLabel', channels_ordered);
    
    title(sprintf('%s (p < %.2f)', conditions{c}, sig_threshold));
    xlabel('Subject B Channels');
    ylabel('Subject A Channels');
end

% save('hyperSession4_maxSurrs1_alpha.mat');

% to-do:
% Run processing on Beta frequancy band
% Split based on speaking turns