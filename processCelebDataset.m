close all; clc ; clear variables;
actors=dir('voxceleb1/voxceleb1_txt/*');
A=size(actors,1);
actors=actors(3:A);
A=size(actors,1);

for i=1:A
    if exist(strcat('features/',actors(i).name,'.mat'),'file') == 0
        disp(['Getting features for: ' actors(i).name '...']);
        samples=getSamplesForActor(actors(i).name);
        features=normalize(getFeaturesForUser(samples));
        save(strcat('features/',actors(i).name,'.mat'),'features');
        clear samples;
    end
end

all_features=dir('features/*');
A=size(all_features,1);
all_features=all_features(3:A); 
A=size(all_features,1);
for i=1:A
    if exist(strcat('models/',actors(i).name,'.model'),'file') == 0
        disp(['Training model for: ' actors(i).name '...']);
        load(strcat('features/',actors(i).name,'.mat'),'features');
        positive_features=features;
        if size(positive_features,1)>0
            negative_features=[];
            for j=randsample([1:(i-1),(i+1):A],round(0.05*A))
                load(strcat('features/',all_features(j).name),'features')
                negative_features=vertcat(negative_features,features);
            end
            positive_features=[positive_features,ones(size(positive_features,1),1)];
            negative_features=[negative_features,-ones(size(negative_features,1),1)];
            [model,type,accuracy]=trainModelForUser(positive_features,negative_features);
            disp(['Trained a ' type ' model for ' actors(i).name ' with an accuracy of ' num2str(accuracy*100) '% !'])
            save(strcat('models/',actors(i).name,'.model'),'model','type','accuracy');
        end
        clear positive_features negative_features;
    end
end