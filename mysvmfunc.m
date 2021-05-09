%% Function for detailed performance comparison of SVM using different type kernel, BoxConstraint for both training and testing data
%% First argument options: 1: BoxConstraint = 0.2, 2: BoxConstraint = 10, 3: BoxConstraint = 1 , 
%% Second argument options: 1: radial basis function kernel, 2: linear kernel, 3: polynomial function of order 2, 4: polynomial function of order 3.
%% Third argument options: 1: training data, 2: testing data


function [f] = mysvmfunc(box,ktype,datatype)
    load P.mat P; 
    load T.mat T;
    %% Dividing feature and target data set to train and test
    [trainP,~,testP,trainInd,valInd,testInd] = dividerand(P,0.7,0,0.3);
    [trainT,~,testTarget] = divideind(T,trainInd,valInd,testInd);
    
    
    if(ktype == 1) % Calculating prediction performance for RBF kernel type
        
        if(box == 1)
            DISCR_svm = fitcsvm(trainP',trainT','Standardize',true,'KernelFunction','RBF', 'KernelScale','auto','BoxConstraint',2e-1); % Using BoxConstraint = .2
        elseif(box == 3)
            DISCR_svm = fitcsvm(trainP',trainT','Standardize',true,'KernelFunction','RBF', 'KernelScale','auto'); % Using default BoxConstraint = 1
        elseif(box == 2)
            DISCR_svm = fitcsvm(trainP',trainT','Standardize',true,'KernelFunction','RBF', 'KernelScale','auto','BoxConstraint',10); % Using BoxConstraint = 10
        end
        if(datatype == 1) % Calculating prediction performance for training data
            TRNpred=predict(DISCR_svm,trainP');
            c = confusionmat(TRNpred,trainT');
            acc = ((c(1,1) + c(2,2))/sum(c,'all'))*100;
            f = "The Training accuracy % of SVM using RBF kernel is:  " + num2str(acc);
            figure;
            confusionchart(c);
            savefig('RBF_train.fig');
           
        elseif(datatype == 2) % Calculating prediction performance for testing data
            TSTpred=predict(DISCR_svm,testP');
            c = confusionmat(TSTpred,testTarget');
            acc = ((c(1,1) + c(2,2))/sum(c,'all'))*100;
            f = "The Testing accuracy % of SVM using RBF kernel is:  " + num2str(acc);
            figure;
            confusionchart(c);
            savefig('RBF_test.fig');
            
        end
    
    elseif(ktype == 2) % Calculating prediction performance for linear kernel type
        
        if(box == 1)
            DISCR_svm = fitcsvm(trainP',trainT','Standardize',true,'KernelFunction','linear', 'KernelScale','auto','BoxConstraint',2e-1);
        elseif(box == 3)
            DISCR_svm = fitcsvm(trainP',trainT','Standardize',true,'KernelFunction','linear', 'KernelScale','auto');
        elseif(box == 2)
            DISCR_svm = fitcsvm(trainP',trainT','Standardize',true,'KernelFunction','linear', 'KernelScale','auto','BoxConstraint',10);
        end
        if(datatype == 1) % Calculating prediction performance for training data
            TRNpred=predict(DISCR_svm,trainP');
            c = confusionmat(TRNpred,trainT');
            acc = ((c(1,1) + c(2,2))/sum(c,'all'))*100;
            f = "The Training accuracy % of SVM using linear kernel is:  " + num2str(acc);
            figure;
            confusionchart(c);
            savefig('linear_train.fig');
            
        elseif(datatype == 2) % Calculating prediction performance for testing data
            TSTpred=predict(DISCR_svm,testP');
            c = confusionmat(TSTpred,testTarget');
            acc = ((c(1,1) + c(2,2))/sum(c,'all'))*100;
            f = "The Testing accuracy % of SVM using linear kernel is:  " + num2str(acc);
            figure;
            confusionchart(c);
            savefig('linear_test.fig');
            
        end
        
    elseif(ktype == 3) % Calculating prediction performance for second order polynomial kernel type
        
        if(box == 1)
            DISCR_svm = fitcsvm(trainP',trainT','Standardize',true,'KernelFunction','Polynomial', 'PolynomialOrder',2, 'KernelScale','auto','BoxConstraint',2e-1);
        elseif(box == 3)
            DISCR_svm = fitcsvm(trainP',trainT','Standardize',true,'KernelFunction','Polynomial', 'PolynomialOrder',2, 'KernelScale','auto');
        elseif(box == 2)
            DISCR_svm = fitcsvm(trainP',trainT','Standardize',true,'KernelFunction','Polynomial', 'PolynomialOrder',2, 'KernelScale','auto','BoxConstraint',10);
        end
        if(datatype == 1) % Calculating prediction performance for training data
            TRNpred=predict(DISCR_svm,trainP');
            c = confusionmat(TRNpred,trainT');
            acc = ((c(1,1) + c(2,2))/sum(c,'all'))*100;
            f = "The Training accuracy % of SVM using second order polynomial kernel is:  " + num2str(acc);
            figure;
            confusionchart(c);
            savefig('2ndpoly_train.fig');
        elseif(datatype == 2) % Calculating prediction performance for testing data
            TSTpred=predict(DISCR_svm,testP');
            c = confusionmat(TSTpred,testTarget');
            acc = ((c(1,1) + c(2,2))/sum(c,'all'))*100;
            f = "The Testing accuracy % of SVM using second order polynomial kernel is:  " + num2str(acc);
            figure;
            confusionchart(c);
            savefig('2ndpoly_test.fig');
            
        end
        
    elseif(ktype == 4)  % Calculating prediction performance for third order polynomial kernel type
        
        if(box == 1)
            DISCR_svm = fitcsvm(trainP',trainT','Standardize',true,'KernelFunction','Polynomial', 'PolynomialOrder',3, 'KernelScale','auto','BoxConstraint',2e-1);
        elseif(box == 3)
            DISCR_svm = fitcsvm(trainP',trainT','Standardize',true,'KernelFunction','Polynomial', 'PolynomialOrder',3, 'KernelScale','auto');
        elseif(box == 2)
            DISCR_svm = fitcsvm(trainP',trainT','Standardize',true,'KernelFunction','Polynomial', 'PolynomialOrder',3, 'KernelScale','auto','BoxConstraint',10);
        end
        if(datatype == 1) % Calculating prediction performance for training data
            TRNpred=predict(DISCR_svm,trainP');
            c = confusionmat(TRNpred,trainT');
            acc = ((c(1,1) + c(2,2))/sum(c,'all'))*100;
            f = "The Training accuracy % of SVM using third order polynomial kernel is:  " + num2str(acc);
            figure;
            confusionchart(c);
            savefig('3rdpoly_train.fig');
            
        elseif(datatype == 2) % Calculating prediction performance for testing data
            TSTpred=predict(DISCR_svm,testP');
            c = confusionmat(TSTpred,testTarget');
            acc = ((c(1,1) + c(2,2))/sum(c,'all'))*100;
            f = "The Testing accuracy % of SVM using third order polynomial kernel is:  " + num2str(acc);
            figure;
            confusionchart(c);
            savefig('3rdpoly_test.fig');
       end
                     
    end
    
close all;
end