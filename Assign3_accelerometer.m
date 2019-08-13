%average
root_directory="572/Data_Mining/";
time_directory=root_directory+"/groundTruth/";
data_directory=root_directory+"/MyoData/";
utensil="spoon";
%list of 20 users
users=["user10","user11","user12","user13","user14","user16","user17","user18"...
       "user19","user21","user22","user23","user24","user25","user26","user27"...
       "user28","user29","user30","user31"];
dwt_data = zeros(20,3); % prepare a data structure to save dwt
non_dwt_data = zeros(20,3); % prepare a data structure to save dwt for non eating
avg_data = zeros(20,3);
non_avg_data = zeros(20,3);
fft_data = zeros(20,3);
non_fft_data = zeros(20,3);
rms_data = zeros(20,3);
non_rms_data = zeros(20,3);
std_data = zeros(20,3);
non_std_data = zeros(20,3);
for index=1:length(users)% assign3
    time_dir=time_directory+users(index)+"/"+utensil+"/";
    files_in_dir = dir(time_dir+"*.txt"); %find a file named with certain patterns
    time_data_matrix = csvread(time_dir +files_in_dir(1).name);
    filename = erase(files_in_dir(1).name,".txt");
    raw_data_matrix = csvread(data_directory+users(index)+"/"+utensil+"/"+filename+"_IMU.txt");

    disp(users(index)+ " for file " + filename);
    i=index;
    %convert the row identififers in groundTruth(time_data_matrix) for raw_data_matrix
    [starting_row,ending_row]= convert_t(time_data_matrix(1,1:2));%use formula given by professor


    result_x = raw_data_matrix(starting_row:ending_row,6);
    dwt_result = dwt(result_x,'sym4');
    dwt_data(i,1) = mean(dwt_result);
    avg_data(i,1) = mean(result_x);
    fft_result = fft(fftshift(result_x));
    fft_data(i,1) = mean(fft_result);
    rms_data(i,1) = rms(result_x);
    std_data(i,1) = std(result_x);

    result_y = raw_data_matrix(starting_row:ending_row,7);
    dwt_result = dwt(result_y,'sym4');
    dwt_data(i,2) = mean(dwt_result);
    avg_data(i,2) = mean(result_y);
    fft_result = fft(fftshift(result_y));
    fft_data(i,2) = mean(fft_result);
    rms_data(i,2) = rms(result_y);
    std_data(i,2) = std(result_y);

    result_z = raw_data_matrix(starting_row:ending_row,8);
    dwt_result = dwt(result_z,'sym4');
    dwt_data(i,3) = mean(dwt_result);
    avg_data(i,3) = mean(result_z);
    fft_result = fft(fftshift(result_z));
    fft_data(i,3) = mean(fft_result);
    rms_data(i,3) = rms(result_z);
    std_data(i,3) = std(result_z);

    % fetch non eating action
    temp=time_data_matrix(1,1:2);
    temp2=time_data_matrix(2,1:2);
    non_eat_start=temp(2)+1;
    non_eat_end=temp2(1)-1;
    [starting_row,ending_row]= convert_t([non_eat_start,non_eat_end]);%use formula given by professor


    result_x = raw_data_matrix(starting_row:ending_row,6);
    dwt_result = dwt(result_x,'sym4');
    non_dwt_data(i,1) = mean(dwt_result);
    non_avg_data(i,1) = mean(result_x);
    fft_result = fft(fftshift(result_x));
    non_fft_data(i,1) = mean(fft_result);
    non_rms_data(i,1) = rms(result_x);
    non_std_data(i,1) = std(result_x);

    result_y = raw_data_matrix(starting_row:ending_row,7);
    dwt_result = dwt(result_y,'sym4');
    non_dwt_data(i,2) = mean(dwt_result);
    non_avg_data(i,2) = mean(result_y);
    nfft_result = fft(fftshift(result_y));
    non_fft_data(i,2) = mean(fft_result);
    non_rms_data(i,2) = rms(result_y);
    non_std_data(i,2) = std(result_y);

    result_z = raw_data_matrix(starting_row:ending_row,8);
    dwt_result = dwt(result_z,'sym4');
    non_dwt_data(i,3) = mean(dwt_result);
    non_avg_data(i,3) = mean(result_z);
    fft_result = fft(fftshift(result_z));
    non_fft_data(i,3) = mean(fft_result);
    non_rms_data(i,3) = rms(result_z);
    non_std_data(i,3) = std(result_z);

    fft_data=real(fft_data); % remove imaginary numbers
    non_fft_data=real(non_fft_data);
    
end
disp("Getting dwt of ");
dwt_graph=computeML("Discrete Wavelet",dwt_data,non_dwt_data);
disp("Getting avg");
avg_graph=computeML("Average",avg_data,non_avg_data);
disp("Getting fft");
fft_graph=computeML("Fast Fourier Transform",fft_data,non_fft_data);
disp("Getting rms");
rms_graph=computeML("Root Mean Square",rms_data,non_rms_data);
disp("Getting std");
std_graph=computeML("Standard Deviation",std_data,non_std_data);
fprintf("break")
function figure1=computeML(feature,eating_data,non_eating_data)
global utensil
global users
global index
[eat_coeff,eat_score,eat_latent] = pca(eating_data);
[non_coeff,non_score,non_latent] = pca(non_eating_data);
pca_eat_train=eat_score(1:12,:); %select 60% of raw data for training
pca_eat_test=eat_score(13:20,:); %select 40% for testing
pca_non_eat_train=non_score(1:12,:);
pca_non_eat_test=non_score(13:20,:);
class_label_train=zeros(24,1); % prepare class label.
class_label_train(1:12)=1; % set the first 12 columns to 1, which we call eating action

class_label_test=zeros(16,1); % prepare class label.
class_label_test(1:8)=1; % set the first 8 columns to 1, which we call eating action
bar_graph_data=[];
% SVM
[confus_precision,confus_recall,confus_f1score]= computeSupportVector(pca_eat_train,pca_non_eat_train,pca_eat_test,pca_non_eat_test,class_label_train,class_label_test);

fprintf('Using SVM, training 24 %s users for %s on feature "%s" produces\n', utensil,users(index),feature);
fprintf('\tPrecision %-9.2f\n', confus_precision);
fprintf('\tRecall %-9.2f\n', confus_recall);
fprintf('\tF1-Score %-9.2f\n', confus_f1score);
bar_graph_data=[bar_graph_data;confus_precision,confus_recall,confus_f1score];
% DT
[confus_precision,confus_recall,confus_f1score]= computeDecisionTree(pca_eat_train,pca_non_eat_train,pca_eat_test,pca_non_eat_test,class_label_train,class_label_test);

fprintf('Using DT, training 24 %s users for %s on feature "%s" produces\n', utensil,users(index),feature);
fprintf('\tPrecision %-9.2f\n', confus_precision);
fprintf('\tRecall %-9.2f\n', confus_recall);
fprintf('\tF1-Score %-9.2f\n', confus_f1score);
bar_graph_data=[bar_graph_data;confus_precision,confus_recall,confus_f1score];

% NN

[confus_precision,confus_recall,confus_f1score]=computeNeuralNet(pca_eat_train,pca_non_eat_train,pca_eat_test,pca_non_eat_test);
fprintf('Using NN, training 24 %s users for %s on feature "%s" produces\n', utensil,users(index),feature);
fprintf('\tPrecision %-9.2f\n', confus_precision);
fprintf('\tRecall %-9.2f\n', confus_recall);
fprintf('\tF1-Score %-9.2f\n', confus_f1score);
bar_graph_data=[bar_graph_data;confus_precision,confus_recall,confus_f1score];

figure1=figure();
catergories = categorical({'SVM','DT','NN'});
graph1=bar(catergories,bar_graph_data);
title_graph=sprintf("%s feature",feature);
title(title_graph);
xlabel('Machine learning algorithm');
ylabel('Percentage');
legend(graph1,{'Precision','Recall','F1'},'Location','eastoutside');
ylim([0 1])
end

function [confus_precision,confus_recall,confus_f1score]= computeSupportVector(pca_eat_train,pca_non_eat_train,pca_eat_test,pca_non_eat_test,class_label_train,class_label_test)
SVMModel = fitcsvm([pca_eat_train;pca_non_eat_train],class_label_train,'Standardize',true,'KernelFunction','RBF',...
    'KernelScale','auto');
prediction = predict(SVMModel,[pca_eat_test;pca_non_eat_test]);
C = confusionmat(class_label_test,prediction);
%DisplayConfusionMatrix(C) % this is to help us find the true/false values
%in confdusion matrix

TP=C(2,2); % true positive to respect of eating action, or class 1
TN=C(1,1);
FP=C(1,2);
FN=C(2,1);
confus_precision= precision(TP,FP);
confus_recall= recall(TP,FN);
confus_f1score=f1score(confus_precision,confus_recall);

end
function [confus_precision,confus_recall,confus_f1score]= computeDecisionTree(pca_eat_train,pca_non_eat_train,pca_eat_test,pca_non_eat_test,class_label_train,class_label_test)
DTModel = fitctree([pca_eat_train;pca_non_eat_train],class_label_train);
prediction = predict(DTModel,[pca_eat_test;pca_non_eat_test]);
C = confusionmat(class_label_test,prediction);
%DisplayConfusionMatrix(C) % this is to help us find the true/false values
%in confdusion matrix

TP=C(2,2); % true positive to respect of eating action, or class 1
TN=C(1,1);
FP=C(1,2);
FN=C(2,1);
confus_precision= precision(TP,FP);
confus_recall= recall(TP,FN);
confus_f1score=f1score(confus_precision,confus_recall);
end
function [confus_precision,confus_recall,confus_f1score]= computeNeuralNet(pca_eat_train,pca_non_eat_train,pca_eat_test,pca_non_eat_test)
% https://www.mathworks.com/help/deeplearning/examples/wine-classification.html
setdemorandstream(391418381) % set seed to a stable random number
net = patternnet(10); % give our network 10 nodes/layers
NN_training_data = [pca_eat_train;pca_non_eat_train];
NN_training_data=transpose(NN_training_data);
NN_testing_data = [pca_eat_test;pca_non_eat_test];
NN_testing_data=transpose(NN_testing_data);

NN_class_label_train=zeros(2,24); % prepare class label.
NN_class_label_train(1,1:12)=1; 
NN_class_label_train(2,13:24)=1; 

net.trainParam.showWindow = false; % https://www.mathworks.com/matlabcentral/answers/301418-how-to-stop-the-nntraintool-window-from-popping-up-whenever-i-run-trainautoencoder
[net,tr] = train(net,NN_training_data,NN_class_label_train);
%nntraintool % this is just for visuals
prediction_float = net(NN_testing_data);
prediction = vec2ind(prediction_float);
NN_class_label_test=ones(1,16); % prepare class label for neural network
NN_class_label_test(9:16)=2; % set the last 8 columns to 2, which NN calls "non eating"


C = confusionmat(NN_class_label_test,prediction);
%DisplayConfusionMatrix(C) % this is to help us find the true/false values
%in confdusion matrix

TP=C(1,1); % true positive to respect of eating action, or class 1
TN=C(2,2);
FP=C(2,1);
FN=C(1,2);
confus_precision= precision(TP,FP);
confus_recall= recall(TP,FN);
confus_f1score=f1score(confus_precision,confus_recall);
end
function [starting_row,ending_row]= convert_t(row)
    %in groundTruth, we have two values, x and y. To convert this into a
    %row for data in MyoData, multiply x by 50 and then divide by 30
    starting_row = round((row(1)*50)/30);
    ending_row = round((row(2)*50)/30);
end

function confus_precision= precision(TP,FP)
confus_precision= TP / (TP+FP);
end
function confus_recall= recall(TP,FN)
confus_recall= TP / (TP+FN);
end
function confus_f1score=f1score(Pre,Rec)
confus_f1score = (2 * Pre * Rec) / (Pre + Rec);
end

function DisplayConfusionMatrix(confMat)
% this function is based off of https://www.mathworks.com/matlabcentral/answers/313397-how-to-plot-confusion-matrix-of-ecoc-classifier
% Display the confusion matrix in a formatted table.
% Convert confusion matrix into percentage form
confMat = bsxfun(@rdivide,confMat,sum(confMat,2));
digits = '0':'1';
colHeadings = arrayfun(@(x)sprintf('%d',x),0:1,'UniformOutput',false);
format = repmat('%-9s',1,11);
header = sprintf(format,'Actual\Predict  |',colHeadings{:});
fprintf('\n%s\n%s\n',header,repmat('-',size(header)));
for idx = 1:numel(digits)
    fprintf('%-9s',   [digits(idx) '               |']);
    fprintf('%-9.2f', confMat(idx,:));
    fprintf('\n')
end
end