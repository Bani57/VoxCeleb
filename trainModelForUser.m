function [model,type,accuracy]=trainModelForUser(positive_features,negative_features)
P=size(positive_features,1);
N=size(negative_features,1);
train_set=vertcat(positive_features(1:round(0.7*P),:),negative_features(1:round(0.7*N),:));
test_set=vertcat(positive_features((round(0.7*P)+1):P,:),negative_features((round(0.7*N)+1):N,:));

num_features=size(positive_features,2);
min_cost=inf;

for K=1:50
    knn_model=fitcknn(train_set(:,1:(num_features-1)),train_set(:,num_features));
    knn_model.Distance='cosine';
    knn_model.NumNeighbors=K;
    knn_model.BreakTies='nearest';
    output = predict(knn_model,test_set(:,1:(num_features-1)));
    [cost,model_accuracy]=costFunction(output,test_set(:,num_features));
    if cost<min_cost
        min_cost=cost;
        best_model=knn_model;
        type='KNN';
        accuracy=model_accuracy;
    end
end

for M=5:5:50
    net=feedforwardnet(M,'trainlm');
    net.trainParam.showWindow = 0;
    net.layers{1}.transferFcn = 'tansig';
    net.layers{2}.transferFcn = 'tansig';
    net = train(net, train_set(:,1:(num_features-1))',train_set(:,num_features)');
    output = sign(sim(net, test_set(:,1:(num_features-1))')');
    [cost,model_accuracy]=costFunction(output,test_set(:,num_features));
    if cost<min_cost
        min_cost=cost;
        best_model=net;
        type='NN';
        accuracy=model_accuracy;
    end
end

for order=1:10
    svm_model=fitcsvm(train_set(:,1:(num_features-1)),train_set(:,num_features),'KernelFunction','polynomial','PolynomialOrder',order,'ClassNames',[-1,1]);
    [output,~] = predict(svm_model,test_set(:,1:(num_features-1)));
    [cost,model_accuracy]=costFunction(output,test_set(:,num_features));
    if cost<min_cost
        min_cost=cost;
        best_model=svm_model;
        type='SVM';
        accuracy=model_accuracy;
    end
end

model=best_model;
end