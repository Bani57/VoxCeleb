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

if exist('VAD/VAD_features.mat','file') == 0
    
    actors=dir('voxceleb1/voxceleb1_txt/*');
    A=size(actors,1);
    actors=actors(3:A);
    A=size(actors,1);
    
    positive_features=zeros(600,C+1,1,2*A);
    positive_features(:,C+1,1,:)=ones(600,1,1,2*A);
    negative_features=zeros(600,C+1,1,2*A);
    negative_features(:,C+1,1,:)=-ones(600,1,1,2*A);
    
    for i=1:A
        disp(['Getting features for: ',actors(i).name,'...']);
        sample_files=dir(strcat('voxceleb1/voxceleb1_txt/',actors(i).name,'/*.wav'));
        if ~isempty(sample_files)
            samplefile_random=sample_files(randsample(1:length(sample_files),1));
            [sample,fs]=audioread(strcat('voxceleb1/voxceleb1_txt/',actors(i).name,'/',samplefile_random.name));
            sample=sample(:,1);
            [ MFCCs, ~, ~ ] = mfcc( sample+randn(size(sample))*1E-10, fs, Tw, Ts, alpha, hamming, R, M, C, L );
            positive_features(:,1:C,1,i)=imresize(MFCCs',[600 C]);
            clear MFCCs sample;

            samplefile_random=sample_files(randsample(length(sample_files),1));
            clear sample_files;
            [sample,fs]=audioread(strcat('voxceleb1/voxceleb1_txt/',actors(i).name,'/',samplefile_random.name));
            clear samplefile_random;
            sample=normalize(sample(:,1));
            for j=1:length(sample)
                sample(j)=sample(j) + 0.2*rand - 0.1;
            end
            [ MFCCs, ~, ~ ] = mfcc( sample + randn(size(sample))*1E-10, fs, Tw, Ts, alpha, hamming, R, M, C, L );
            positive_features(:,1:C,1,i+A)=imresize(MFCCs',[600 C]);
    
            sample_random_noise=2*rand(1,length(sample)) - 1;
            [ MFCCs, ~, ~ ] = mfcc( sample_random_noise+randn(size(sample_random_noise))*1E-10, fs, Tw, Ts, alpha, hamming, R, M, C, L );
            negative_features(:,1:C,1,i)=imresize(MFCCs',[600 C]);
            clear sample_random_noise;
        
            sample_random_noise_pulses=zeros(1,length(sample));
            breaks=sort(randsample(1:length(sample),30));
            for j=1:2:30
                sample_random_noise_pulses(breaks(j):breaks(j+1))=2*rand(1,breaks(j+1)-breaks(j)+1)-1;
            end
            [ MFCCs, ~, ~ ] = mfcc( sample_random_noise_pulses+randn(size(sample_random_noise_pulses))*1E-10, fs, Tw, Ts, alpha, hamming, R, M, C, L );
            negative_features(:,1:C,1,i+A)=imresize(MFCCs',[600 C]);
            clear sample_random_noise_pulses sample;
        end
    end
    
    dataset=zeros(600,C+1,1,4*A);
    dataset(:,:,1,1:2*A)=positive_features;
    dataset(:,:,1,(2*A+1):4*A)=negative_features;
    clear positive_features negative_features;
    ind = find(dataset(:,1:C,1,:)~=zeros(600,C,1,1));
    [~, ~, ~, i4] = ind2sub(size(dataset(:,1:C,1,:)), ind);
    dataset=dataset(:,:,1,unique(i4));
    dataset=dataset(:,:,1,randperm(size(dataset,4)));
    save('VAD/VAD_features.mat','dataset');
else
    load('VAD/VAD_features.mat','dataset');
end

N=size(dataset,4);
train_set=dataset(:,1:C,1,1:round(0.7*N));
test_set=dataset(:,1:C,1,(round(0.7*N)+1):N);
train_classes=permute(dataset(1,C+1,1,1:round(0.7*N)),[4 1 2 3]);
test_classes=permute(dataset(1,C+1,1,(round(0.7*N)+1):N),[4 1 2 3]);
clear dataset;
    
if exist('VAD/VAD.model','file') == 0

    layers = [
        imageInputLayer([600 C 1])  
        convolution2dLayer(5,40,'Stride',2,'Padding','same')
        reluLayer
        convolution2dLayer(5,20,'Stride',2,'Padding','same')
        reluLayer
        convolution2dLayer(5,10,'Stride',2,'Padding','same')
        reluLayer
        averagePooling2dLayer([100 1],'Stride',2,'Padding','same')
        dropoutLayer(0.25)
        fullyConnectedLayer(2)
        softmaxLayer
        classificationLayer];
    
    maxEpochs=12;
    miniBatchSize=round(size(train_set,4)/975);
    valid_set=train_set(:,:,1,(round(0.9*size(train_set,4))+1):size(train_set,4));
    valid_classes=train_classes((round(0.9*length(train_classes))+1):length(train_classes));
    train_set=train_set(:,:,1,1:round(0.9*size(train_set,4)));
    train_classes=train_classes(1:round(0.9*length(train_classes)));
    
    options = trainingOptions('sgdm','MiniBatchSize',miniBatchSize,'MaxEpochs',maxEpochs,'ValidationData',{valid_set,categorical(valid_classes)},'ValidationFrequency',20,'LearnRateSchedule','piecewise','InitialLearnRate',0.001,'LearnRateDropFactor',0.1,'LearnRateDropPeriod',4,'Plots','training-progress');
    trainedNet = trainNetwork(train_set,categorical(train_classes),layers,options);
    save('VAD/VAD.model','trainedNet');
else
    load('VAD/VAD.model','-mat','trainedNet');
end

[output,~]=classify(trainedNet,test_set);
output=double(output);
output(output==1)=-1;
output(output==2)=1;
[cost,accuracy]=costFunction(output,test_classes);
disp(['Cost = ',num2str(cost)]);
disp(['Accuracy = ',num2str(accuracy)]);

if exist('VAD/VAD2.model','file') == 0
    
    actors=dir('voxceleb1/voxceleb1_txt/*');
    A=size(actors,1);
    actors=actors(3:A);
    A=size(actors,1);
    positive_features=zeros(2*A,21);
    positive_features(:,21)=ones(2*A,1);

    for i=1:A
        disp(['Getting features for: ',actors(i).name,'...']);
        load(strcat('features/',actors(i).name,'.mat'),'features');
        if size(features,1)>0
            random_features=features(randsample(1:size(features,1),2),:);
            positive_features((2*i-1):(2*i),1:20)=random_features;
            clear features random_features;
        end
    end
    ind = find(positive_features(:,1:20)~=zeros(1,20));
    [i1,~] = ind2sub(size(positive_features(:,1:20)), ind);
    positive_features=positive_features(unique(i1),:);
    
    
    disp('Getting features of noise samples...');
    pause(1)
    noises=dir('ESC-50-master/audio/*');
    B=size(noises,1);
    noises=noises(3:B);
    B=size(noises,1);
    negative_features=zeros(B,21);
    negative_features(:,21)=-ones(B,1);
    sample_cell=cell(B,2);
    
    for i=1:B
        [noise_sample,fs]=audioread(strcat('ESC-50-master/audio/',noises(i).name));
        sample_cell{i,1}=noise_sample(:,1);
        sample_cell{i,2}=fs;
        clear noise_sample;
    end
    negative_features(:,1:20)=normalize(getFeaturesForUser(sample_cell));
    
    disp('Training the VAD model...');
    P=size(positive_features,1);
    N=size(negative_features,1);
    train_set=vertcat(positive_features(1:round(0.7*P),:),negative_features(1:round(0.7*N),:));
    test_set=vertcat(positive_features((round(0.7*P)+1):P,:),negative_features((round(0.7*N)+1):N,:));
    num_features=size(positive_features,2);
    min_cost=inf;
    for M=1:50
        net=feedforwardnet(M,'trainlm');
        net.layers{1}.transferFcn = 'tansig';
        net.layers{2}.transferFcn = 'tansig';
        net = train(net, train_set(:,1:(num_features-1))',train_set(:,num_features)');
        output = sign(sim(net, test_set(:,1:(num_features-1))')');
        [cost,model_accuracy]=costFunction(output,test_set(:,num_features));
        if cost<min_cost
            min_cost=cost;
            best_model=net;
            accuracy=model_accuracy;
        end
    end
    model=best_model;
    
    disp(['Trained a neural network model for noise detection with an accuracy of ' num2str(accuracy*100) '% !'])
    save('VAD/VAD2.model','model','accuracy');
end