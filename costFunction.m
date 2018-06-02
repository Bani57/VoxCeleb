function [cost,accuracy]=costFunction(output,target)
FP = sum(and(target==-1,output==1));
FN = sum(and(target==1,output==-1));
TP = sum(and(target==1,output==1));
TN = sum(and(target==-1,output==-1));
cost_TP=0;
cost_TN=0;
cost_FP=1000;
cost_FN=10;
cost=FP*cost_FP + FN*cost_FN + TP*cost_TP + TN*cost_TN;
accuracy=(TP+TN)/(TP+FP+FN+TN);
precision_positive=TP/(TP+FP);
precision_negative=TN/(TN+FN);
recall_positive=TP/(TP+FN);
recall_negative=TN/(TN+FP);
F_positive=2/(1/precision_positive+1/recall_positive);
F_negative=2/(1/precision_negative+1/recall_negative);
F_measure=(F_positive+F_negative)/2;
cost=cost/F_measure;
end