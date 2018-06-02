clear variables; clc; close all;

Tw = 25;           % analysis frame duration (ms)
Ts = 12.5;         % analysis frame shift (ms)
alpha = 0.97;      % preemphasis coefficient
R = [ 300 8000 ];  % frequency range to consider
M = 40;            % number of filterbank channels 
C = 100;           % number of cepstral coefficients
L = 22;            % cepstral sine lifter parameter
% hamming window (see Eq. (5.2) on p.73 of [1])
hamming = @(N)(0.54-0.46*cos(2*pi*(0:N-1).'/(N-1)));
Nw = round( 1E-3*Tw*44100 );    % frame duration (samples)
Ns = round( 1E-3*Ts*44100 );    % frame shift (samples)

disp('Welcome to the register page!');
username = input('Enter username of new user: ','s');
if exist(strcat('models/',username,'.model'),'file') ~= 0
    disp('That user account already exists!');
else
    disp(['OK ',username,', you will have to submit an audio sample of your voice with a length of about 1 minute.']);
    pause(1)
    recorder=audiorecorder(44100,16,1);
    disp('Prepare to start speaking...');
    pause(1)
    disp('3...');
    pause(1)
    disp('2...');
    pause(1)
    disp('1...');
    pause(1)
    disp('Start speaking!');
    recordblocking(recorder, randsample(60:90,1));
    disp('OK, we are done.');
    load gong
    sound(y,Fs)
    pause(1)
    sample = getaudiodata(recorder);
    sample = sample(:,1);
    disp('Getting features from your sample...');
    pause(1)
    [ MFCCs, ~, ~ ] = mfcc( sample+randn(size(sample))*1E-10, 44100, Tw, Ts, alpha, hamming, R, M, C, L );
    features=zeros(600,C,1,1);
    features(:,:,1,1)=imresize(MFCCs',[600 C]);
    load('VAD/VAD.model','-mat','trainedNet');
    [output,~]=classify(trainedNet,features);
    output=double(output);
    if output==1
        output=-1;
    elseif output==2
        output=1;
    end
    [frames,~] = vec2frames( sample, Nw, Ns, 'cols', hamming, false );
    clear sample;
    F=size(frames,2);
    frame_samples=cell(F,2);
    for i=1:F
        frame_samples{i,1}=frames(:,i);
        frame_samples{i,2}=44100;
    end
    clear frames;
    positive_features=normalize(getFeaturesForUser(frame_samples));
    clear frame_samples;
    features=positive_features;
    load('VAD/VAD2.model','-mat','model');
    probability=sim(model, mean(features)')';
    output2 = sign(probability);
    disp(['Probability that the sample was not noisy: ',num2str(probability)]);
    if output==-1 || output2==-1 || probability<=0.85
        disp('That sample was a bit too noisy, try again!');
    else
        disp('Good sample! Now we can train a model for you!');
        pause(1)
        disp('Getting features from other users...');
        pause(1)
        all_features=dir('features/*');
        A=size(all_features,1);
        all_features=all_features(3:A); 
        A=size(all_features,1);
        negative_features=[];
        for j=randsample(1:A,round(0.05*A))
            load(strcat('features/',all_features(j).name),'features')
            negative_features=vertcat(negative_features,features);
        end
        positive_features=[positive_features,ones(size(positive_features,1),1)];
        negative_features=[negative_features,-ones(size(negative_features,1),1)];
        disp('Training a model for you...');
        [model,type,accuracy]=trainModelForUser(positive_features,negative_features);
        disp(['Trained a ' type ' model for ' username ' with an accuracy of ' num2str(accuracy*100) '% !']);
        save(strcat('features/',username,'.mat'),'features');
        clear features;
        save(strcat('models/',username,'.model'),'model','type','accuracy');
    end
end