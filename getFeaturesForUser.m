function features=getFeaturesForUser(samples)
num_samples=size(samples,1);
features=zeros(num_samples,20);

Tw = 25;           % analysis frame duration (ms)
Ts = 10;           % analysis frame shift (ms)
alpha = 0.97;      % preemphasis coefficient
R = [ 300 3700 ];  % frequency range to consider
M = 20;            % number of filterbank channels 
C = 13;            % number of cepstral coefficients
L = 22;            % cepstral sine lifter parameter

% hamming window (see Eq. (5.2) on p.73 of [1])
hamming = @(N)(0.54-0.46*cos(2*pi*(0:N-1).'/(N-1)));

for i=1:num_samples
    features_sample=zeros(1,18);
    [ MFCCs, ~, ~ ] = mfcc( samples{i,1}+randn(size(samples{i,1}))*1E-10, samples{i,2}, Tw, Ts, alpha, hamming, R, M, C, L );
    features_sample(1:13)=mean(MFCCs,2); % Mean Mel cepstral coefficients
    features_sample(14)=mean(samples{i,1}); % Mean intensity
    features_sample(15)=std(samples{i,1}); % Standard deviation of intensity
    Nw = round(1E-3*Tw*samples{i,2});
    Ns = round(1E-3*Ts*samples{i,2});
    nfft = 2^nextpow2(Nw);      
    frames = vec2frames( samples{i,1}, Nw, Ns, 'cols', hamming, false );
    MAG = abs( fft(frames,nfft,1) );
    MAG = mean(MAG,2);
    maximumMAGs=find(MAG==max(MAG));
    minimumMAGs=find(MAG==min(MAG));
    features_sample(16)=maximumMAGs(1); % Frequency with maximum amplitude
    features_sample(17)=minimumMAGs(1); % Frequency with minimum amplitude
    features_sample(18:20)=getTempo(samples{i,1}); % Tempo statistics
    features(i,:)=features_sample;
end