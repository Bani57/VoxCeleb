clear variables; clc; close all;

disp('Welcome to the login page!');
username = input('Enter username: ','s');

if exist(strcat('models/',username,'.model'),'file') == 0
    disp('That user account does not exist!');
else
    disp(['OK ',username,', you will have to talk for about 20 to 30 seconds in order to prove your identity.']);
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
    recordblocking(recorder, randsample(20:30,1));
    disp('OK, we are done.');
    load gong
    sound(y,Fs)
    pause(1)
    sample = getaudiodata(recorder);
    sample = sample(:,1);
    Tw = 25;            % analysis frame duration (ms)
    Ts = 12.5;            % analysis frame shift (ms)
    Nw = round( 1E-3*Tw*44100 );    % frame duration (samples)
    Ns = round( 1E-3*Ts*44100 );    % frame shift (samples)
    hamming = @(N)(0.54-0.46*cos(2*pi*(0:N-1).'/(N-1)));
    
    disp('Getting features from your sample...');
    pause(1)
    [frames,~] = vec2frames( sample, Nw, Ns, 'cols', hamming, false );
    clear sample;
    F=size(frames,2);
    frame_samples=cell(F,2);
    for i=1:F
        frame_samples{i,1}=frames(:,i);
        frame_samples{i,2}=44100;
    end
    clear frames;
    features=normalize(getFeaturesForUser(frame_samples));
    features=mean(features);
    clear frame_samples;
    
    disp('Getting your model...');
    pause(1)
    load(strcat('models/',username,'.model'),'-mat','model');
    load(strcat('models/',username,'.model'),'-mat','type');
    disp('Classifying...');
    pause(1)
    if strcmp(type,'KNN')
        [output,probability,~] = predict(model,features);
        disp(['Probability: ' num2str(probability(2))]);
    elseif strcmp(type,'NN')
        probability=sim(model, features')';
        disp(['Probability: ' num2str(probability)]);
        output = sign(probability);
    else
        [output,probability] = predict(model,features);
        disp(['Probability: ' num2str(probability(2))]);
    end
    if output==1 && probability>=0.85
        disp(['Welcome, ' username '!']);
    else
        disp(['I don''t think you are ' username '! Try again!']);
    end
end