%average
global utensil
global users
global index
root_directory="572/Data_Mining/";
time_directory=root_directory+"/groundTruth/";
data_directory=root_directory+"/MyoData/";
utensil="spoon";

users=dir(time_directory); %find all folders in groundTruth 
users=users(3:length(users));  %delete the first two records, '.' and '..' which are current and parent directory
index=1;% get first user in users
time_dir=time_directory+users(index).name+"/"+utensil+"/";
files_in_dir = dir(time_dir+"*.txt"); %find a file named with certain patterns
time_data_matrix = csvread(time_dir +files_in_dir(1).name);
filename = erase(files_in_dir(1).name,".txt");

raw_data_matrix = csvread(data_directory+users(index).name+"/"+utensil+"/"+filename+"_IMU.txt");

% write csv headers
% https://www.mathworks.com/matlabcentral/answers/259937-csvwrite-a-matrix-with-header
textHeader = ',SVM,SVM,SVM,DT,DT,DT,NN,NN,NN';
%write header to file
fid = fopen('Assign2_DWT.csv','w'); 
fprintf(fid,'%s\n',textHeader);
fclose(fid);
textHeader = 'user,Precis,Recall,F1,Precis,Recall,F1,Precis,Recall,F1,';
%write header to file
fid = fopen('Assign2_DWT.csv','a'); 
fprintf(fid,'%s\n',textHeader);
fclose(fid);
%copy csv into 4 other files
copyfile 'Assign2_DWT.csv' 'Assign2_AVG.csv'
copyfile 'Assign2_DWT.csv' 'Assign2_FFT.csv'
copyfile 'Assign2_DWT.csv' 'Assign2_RMS.csv'
copyfile 'Assign2_DWT.csv' 'Assign2_STD.csv'

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


for i = 1:20
    %convert the row identififers in groundTruth(time_data_matrix) for raw_data_matrix
    [starting_row,ending_row]= convert_t(time_data_matrix(i,1:2));%use formula given by professor


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
    temp=time_data_matrix(i,1:2);
    temp2=time_data_matrix(i+1,1:2);
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
global MLModels
MLModels=cell(5); % create an array of objects to hold the ML models
% first row is dwt, with col 1=SVM, col 2=DT, and col 3=NN
% second row is avg, etc
disp("Getting dwt of " +users(index).name+ " for file " + filename);
computeML("Assign2_DWT",dwt_data,non_dwt_data,1);
disp("Getting avg");
computeML("Assign2_AVG",avg_data,non_avg_data,2);
disp("Getting fft");
computeML("Assign2_FFT",fft_data,non_fft_data,3);
disp("Getting rms");
computeML("Assign2_RMS",rms_data,non_rms_data,4);
disp("Getting std");
computeML("Assign2_STD",std_data,non_std_data,5);

%%%%%%%%%%% Start the loop to process other users
% we start at index 2 since index 1 was used to create ML models
for index=2:length(users)
    if strcmp(users(index).name,'user9')
        continue; % skip user9 since that does not exist in myodata
    end
    time_dir=time_directory+users(index).name+"/"+utensil+"/";
    files_in_dir = dir(time_dir+"*.txt"); %find a file named with certain patterns
    time_data_matrix = csvread(time_dir +files_in_dir(1).name);
    filename = erase(files_in_dir(1).name,".txt");

    raw_data_matrix = csvread(data_directory+users(index).name+"/"+utensil+"/"+filename+"_IMU.txt");


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



    for i = 1:20
        %convert the row identififers in groundTruth(time_data_matrix) for raw_data_matrix
        [starting_row,ending_row]= convert_t(time_data_matrix(i,1:2));%use formula given by professor


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
        temp=time_data_matrix(i,1:2);
        temp2=time_data_matrix(i+1,1:2);
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
    disp("Getting dwt of " +users(index).name+ " for file " + filename);
    computeML("Assign2_DWT",dwt_data,non_dwt_data,1);
    disp("Getting avg");
    computeML("Assign2_AVG",avg_data,non_avg_data,2);
    disp("Getting fft");
    computeML("Assign2_FFT",fft_data,non_fft_data,3);
    disp("Getting rms");
    computeML("Assign2_RMS",rms_data,non_rms_data,4);
    disp("Getting std");
    computeML("Assign2_STD",std_data,non_std_data,5);
end
function computeML(feature,eating_data,non_eating_data,ind)
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
    bar_graph_data=[users(index).name,"drop"];
    
    
    % SVM
    [confus_precision,confus_recall,confus_f1score]= computeSupportVector(pca_eat_train,pca_non_eat_train,pca_eat_test,pca_non_eat_test,class_label_train,class_label_test,ind);

%     fprintf('Using SVM, training 24 %s actions for %s on feature "%s" produces\n', utensil,users(index).name,feature);
%     fprintf('\tPrecision %-9.2f\n', confus_precision);
%     fprintf('\tRecall %-9.2f\n', confus_recall);
%     fprintf('\tF1-Score %-9.2f\n', confus_f1score);
    bar_graph_data=[bar_graph_data,confus_precision,confus_recall,confus_f1score];
    % DT
    [confus_precision,confus_recall,confus_f1score]= computeDecisionTree(pca_eat_train,pca_non_eat_train,pca_eat_test,pca_non_eat_test,class_label_train,class_label_test,ind);

%     fprintf('Using DT, training 24 %s actions for %s on feature "%s" produces\n', utensil,users(index).name,feature);
%     fprintf('\tPrecision %-9.2f\n', confus_precision);
%     fprintf('\tRecall %-9.2f\n', confus_recall);
%     fprintf('\tF1-Score %-9.2f\n', confus_f1score);
    bar_graph_data=[bar_graph_data,confus_precision,confus_recall,confus_f1score];

    % NN

    [confus_precision,confus_recall,confus_f1score]=computeNeuralNet(pca_eat_train,pca_non_eat_train,pca_eat_test,pca_non_eat_test,ind);
%     fprintf('Using NN, training 24 %s actions for %s on feature "%s" produces\n', utensil,users(index).name,feature);
%     fprintf('\tPrecision %-9.2f\n', confus_precision);
%     fprintf('\tRecall %-9.2f\n', confus_recall);
%     fprintf('\tF1-Score %-9.2f\n', confus_f1score);
    bar_graph_data=[bar_graph_data,confus_precision,confus_recall,confus_f1score];

    % append to csv
    % https://www.mathworks.com/matlabcentral/answers/259937-csvwrite-a-matrix-with-header
    bar_graph_data(2)=[];% drop the temporary string 
    cHeader = string(bar_graph_data);
    commaHeader = [cHeader;repmat({','},1,numel(cHeader))]; %insert commaas
    commaHeader = commaHeader(:)';
    textHeader = cell2mat(commaHeader); %cHeader in text with commas
    %write header to file
    fid = fopen(feature+'.csv','a'); 
    fprintf(fid,'%s\n',textHeader);
    fclose(fid);

end

function [confus_precision,confus_recall,confus_f1score]= computeSupportVector(pca_eat_train,pca_non_eat_train,pca_eat_test,pca_non_eat_test,class_label_train,class_label_test,ind)
    global MLModels
    if isempty(MLModels{ind,1})
        SVMModel = fitcsvm([pca_eat_train;pca_non_eat_train],class_label_train,'Standardize',true,'KernelFunction','RBF',...
        'KernelScale','auto');
         MLModels{ind,1}= SVMModel; %save the model into global array 
    else
        SVMModel=MLModels{ind,1}; % retrieve the SVM model from global array
    end
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
function [confus_precision,confus_recall,confus_f1score]= computeDecisionTree(pca_eat_train,pca_non_eat_train,pca_eat_test,pca_non_eat_test,class_label_train,class_label_test,ind)
    global MLModels
    if isempty(MLModels{ind,2})
         DTModel = fitctree([pca_eat_train;pca_non_eat_train],class_label_train);
         MLModels{ind,2}= DTModel; %save the model into global array 
    else
        DTModel=MLModels{ind,2}; % retrieve the DT model from global array
    end
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
function [confus_precision,confus_recall,confus_f1score]= computeNeuralNet(pca_eat_train,pca_non_eat_train,pca_eat_test,pca_non_eat_test,ind)
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
    global MLModels
    if isempty(MLModels{ind,3})
        [net,tr] = train(net,NN_training_data,NN_class_label_train);
          MLModels{ind,3}=net; %save the model into global array 
    else
        net=MLModels{ind,3}; % retrieve the NN model from global array
    end
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
if isnan(confus_precision)
    confus_precision=0;
end
end
function confus_recall= recall(TP,FN)
confus_recall= TP / (TP+FN);
if isnan(confus_recall)
    confus_recall=0;
end
end
function confus_f1score=f1score(Pre,Rec)
confus_f1score = (2 * Pre * Rec) / (Pre + Rec);
if isnan(confus_f1score)
    confus_f1score=0;
end
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