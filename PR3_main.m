%% Printing the results for SVM using different type kernel, BoxConstraint for both training and testing data
%% Saving all the confusion matrix for different kernel type with train / test data combination


for i = 1:3 % counter for printing values for different BoxConstraint value
    fprintf("\n\n")
    disp("==========================prediction accuracy results for type " + num2str(i) + " BoxConstraint value: ========================")
    fprintf("\n")
    
    for j = 1:4 % counter for printing values for different svm kernel value
        fprintf("\n")
        disp("prediction accuracy results for type " + num2str(j) + " kernel type: ====>")
        fprintf("\n")
        for k = 1:2 % counter for printing values for training and test data
            disp(mysvmfunc(i,j,k));
        
        end
    end
    
end