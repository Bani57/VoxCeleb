clear variables; clc; close all;

load('features/Andrej_Janchevski.mat','features');
figure;
normplot(features)
[~,~]=legend("MFCC 1","MFCC 2","MFCC 3","MFCC 4","MFCC 5","MFCC 6","MFCC 7","MFCC 8","MFCC 9","MFCC 10","MFCC 11","MFCC 12","MFCC 13","Интензитет 1","Интензитет 2","Висина на тон 1","Висина на тон 2","Темпо 1","Темпо 2","Темпо 3");


model_files=dir('models/*.model');
M=size(model_files,1);
model_files=model_files(3:M);
M=size(model_files,1);

KNN_count=0;
NN_count=0;
SVM_count=0;
for i=1:M
    load(strcat('models/',model_files(i).name),'-mat','type');
    if strcmp(type,'KNN')
        KNN_count=KNN_count+1;
    elseif strcmp(type,'NN')
        NN_count=NN_count+1;
    else
        SVM_count=SVM_count+1;
    end
end

KNN_stats=cell(KNN_count,3);
NN_stats=cell(NN_count,3);
SVM_stats=cell(SVM_count,3);
k=1;n=1;s=1;
for i=1:M
    load(strcat('models/',model_files(i).name),'-mat','model');
    load(strcat('models/',model_files(i).name),'-mat','type');
    load(strcat('models/',model_files(i).name),'-mat','accuracy');
    if strcmp(type,'KNN')
        KNN_stats{k,1}=model_files(i).name;
        KNN_stats{k,2}=accuracy;
        KNN_stats{k,3}=model.NumNeighbors;
        k=k+1;
    elseif strcmp(type,'NN')
        NN_stats{n,1}=model_files(i).name;
        NN_stats{n,2}=accuracy;
        NN_stats{n,3}=model.layers{1}.dimensions;
        n=n+1;
    else
        SVM_stats{s,1}=model_files(i).name;
        SVM_stats{s,2}=accuracy;
        SVM_stats{s,3}=model.KernelParameters.Order;
        s=s+1;
    end
end

KNN_stats=sortrows(KNN_stats,2,'descend');
NN_stats=sortrows(NN_stats,2,'descend');
SVM_stats=sortrows(SVM_stats,2,'descend');

disp('*** Top 10 KNN models ***');
for i=1:10
    disp([extractBefore(KNN_stats{i,1},".model"),'    ',num2str(KNN_stats{i,2}*100),'%    ',num2str(KNN_stats{i,3})]);
end
fprintf('\n');
disp('*** Top 10 NN models ***');
for i=1:10
    disp([extractBefore(NN_stats{i,1},".model"),'    ',num2str(NN_stats{i,2}*100),'%    ',num2str(NN_stats{i,3})]);
end
fprintf('\n');
disp('*** Top 10 SVM models ***');
for i=1:10
    disp([extractBefore(SVM_stats{i,1},".model"),'    ',num2str(SVM_stats{i,2}*100),'%    ',num2str(SVM_stats{i,3})]);
end

actors=dir('voxceleb1/voxceleb1_txt/*');
A=size(actors,1);
actors=actors(3:A);
A=size(actors,1);
all_features=dir('features/*');
all_features=all_features(3:(A+2));

load(strcat('models/',KNN_stats{1,1}),'-mat','model');
username=extractBefore(KNN_stats{1,1},".model");
load(strcat('features/',username,'.mat'),'features');
positive_features=features;
negative_features=[];
i=find(strcmp({actors.name}, username)==1);
for j=randsample([1:(i-1),(i+1):A],round(0.05*A))
    load(strcat('features/',all_features(j).name),'features')
    negative_features=vertcat(negative_features,features);
end

%[~,pca_positive,~]=pca(positive_features,'NumComponents',2);
%[~,pca_negative,~]=pca(negative_features,'NumComponents',2);
pca_positive = tsne(positive_features,'Algorithm','barneshut','NumPCAComponents',20);
pca_negative = tsne(negative_features,'Algorithm','barneshut','NumPCAComponents',20);
dataset = vertcat(positive_features,negative_features);
pca_dataset = vertcat(pca_positive,pca_negative);
searcher=ExhaustiveSearcher(dataset,'Distance','cosine');
[n,~] = knnsearch(searcher,positive_features(1,:),'k',model.NumNeighbors+1);

figure;
scatter(pca_positive(:,1),pca_positive(:,2),30,'green','O')
title(strcat("Визуелизација на податоците на KNN моделот за ",username),'Interpreter', 'none')
xlabel("Прва компонента")
ylabel("Втора компонента")
hold on;
scatter(pca_negative(:,1),pca_negative(:,2),30,'red','x')
hold on;
line(pca_positive(1,1),pca_positive(1,2),'color','blue','marker','o','linestyle','none','markersize',10)
hold on;
line(pca_dataset(n(2:model.NumNeighbors+1),1),pca_dataset(n(2:model.NumNeighbors+1),2),'color','black','marker','o','linestyle','none','markersize',10)
[h,~]=legend("Примерок од корисникот","Примерок од друг корисник","Тест примерок","Најблиски соседи");


load(strcat('models/',NN_stats{1,1}),'-mat','model');
username=extractBefore(NN_stats{1,1},".model");
load(strcat('features/',username,'.mat'),'features');
positive_features=features;
negative_features=[];
i=find(strcmp({actors.name}, username)==1);
for j=randsample([1:(i-1),(i+1):A],round(0.05*A))
    load(strcat('features/',all_features(j).name),'features')
    negative_features=vertcat(negative_features,features);
end

pca_positive = tsne(positive_features,'Algorithm','barneshut','NumPCAComponents',20);
pca_negative = tsne(negative_features,'Algorithm','barneshut','NumPCAComponents',20);

figure;
scatter(pca_positive(:,1),pca_positive(:,2),30,'green','O')
title(strcat("Визуелизација на податоците на NN моделот за ",username),'Interpreter', 'none')
xlabel("Прва компонента")
ylabel("Втора компонента")
hold on;
scatter(pca_negative(:,1),pca_negative(:,2),30,'red','x')
[h,~]=legend("Примерок од корисникот","Примерок од друг корисник");

load(strcat('models/',SVM_stats{1,1}),'-mat','model');
username=extractBefore(SVM_stats{1,1},".model");
load(strcat('features/',username,'.mat'),'features');
positive_features=features;
negative_features=[];
i=find(strcmp({actors.name}, username)==1);
for j=randsample([1:(i-1),(i+1):A],round(0.05*A))
    load(strcat('features/',all_features(j).name),'features')
    negative_features=vertcat(negative_features,features);
end

pca_positive = tsne(positive_features,'Algorithm','barneshut','NumPCAComponents',20);
pca_negative = tsne(negative_features,'Algorithm','barneshut','NumPCAComponents',20);
dataset=vertcat(positive_features,negative_features);
sv=model.SupportVectors;

figure;
scatter(pca_positive(:,1),pca_positive(:,2),30,'green','O')
title(strcat("Визуелизација на податоците на SVM моделот за ",username),'Interpreter', 'none')
xlabel("Прва компонента")
ylabel("Втора компонента")
hold on;
scatter(pca_negative(:,1),pca_negative(:,2),30,'red','x')
hold on;
line(sv(:,1),sv(:,2),'color','black','marker','o','linestyle','none','markersize',10)
[h,~]=legend("Примерок од корисникот","Примерок од друг корисник","Придружувачки вектор");